#include "semantic_map_builder_node.h"

namespace semantic_map_builder{

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace lucrezio_logical_camera;

SemanticMapBuilderNode::SemanticMapBuilderNode(ros::NodeHandle nh_):
    SemanticMapBuilder(),
    _nh(nh_),
    _logical_image_sub(_nh,"/gazebo/logical_camera_image",1),
    _depth_cloud_sub(_nh,"/camera/depth/points",1),
    _rgb_image_sub(_nh,"/camera/rgb/image_raw", 1),
    _depth_image_sub(_nh,"/camera/depth/image_raw", 1),
    _synchronizer(FilterSyncPolicy(10),_logical_image_sub,_depth_cloud_sub,_rgb_image_sub,_depth_image_sub),
    _it(_nh){

    _raw_depth_scale = 0.001;
    _camera_height = 1.0f;
    _got_info = false;
    _camera_info_sub = _nh.subscribe("/camera/depth/camera_info",
                                     1000,
                                     &SemanticMapBuilderNode::cameraInfoCallback,
                                     this);

    _synchronizer.registerCallback(boost::bind(&SemanticMapBuilderNode::filterCallback, this, _1, _2, _3, _4));

    _label_image_pub = _it.advertise("/camera/rgb/label_image", 1);

    _shape = visualization_msgs::Marker::CUBE;
    _markers_pub = _nh.advertise<visualization_msgs::MarkerArray>("visualization_markers",1);
}

void SemanticMapBuilderNode::cameraInfoCallback(const CameraInfo::ConstPtr &camera_info_msg){
    CameraInfo camerainfo;
    camerainfo.K = camera_info_msg->K;

    ROS_INFO("Got camera info!");
    _K(0,0) = camerainfo.K.c_array()[0];
    _K(0,1) = camerainfo.K.c_array()[1];
    _K(0,2) = camerainfo.K.c_array()[2];
    _K(1,0) = camerainfo.K.c_array()[3];
    _K(1,1) = camerainfo.K.c_array()[4];
    _K(1,2) = camerainfo.K.c_array()[5];
    _K(2,0) = camerainfo.K.c_array()[6];
    _K(2,1) = camerainfo.K.c_array()[7];
    _K(2,2) = camerainfo.K.c_array()[8];

    cerr << _K << endl;

    _invK = _K.inverse();

    _got_info = true;
    _camera_info_sub.shutdown();
}

constexpr unsigned int str2int(const char* str, int h = 0){
    return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

int SemanticMapBuilderNode::type2id(const string &type){
    switch (str2int(type.c_str())){
    case str2int("table"):
        return 0;
    case str2int("chair"):
        return 1;
    case str2int("bookcase"):
        return 2;
    case str2int("couch"):
        return 3;
    case str2int("cabinet"):
        return 4;
    case str2int("plant"):
        return 5;
    default:
        return 6;
    }
}

void SemanticMapBuilderNode::makeMarkerFromObject(visualization_msgs::Marker &marker,
                                                  const Object &object){
    marker.header.frame_id = "/map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "basic_shapes";
    marker.id = type2id(object.type);
    marker.type = _shape;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = object.centroid.x();
    marker.pose.position.y = object.centroid.y();
    marker.pose.position.z = object.centroid.z();
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = object.size.x();
    marker.scale.y = object.size.y();
    marker.scale.z = object.size.z();

    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration();
}


void SemanticMapBuilderNode::filterCallback(const LogicalImage::ConstPtr &logical_image_msg,
                                            const PointCloudType::ConstPtr &depth_cloud_msg,
                                            const Image::ConstPtr &rgb_image_msg,
                                            const Image::ConstPtr &depth_image_msg){
    if(_got_info && !logical_image_msg->models.empty()){

        ROS_INFO("--------------------------");
        ROS_INFO("Executing filter callback!");
        ROS_INFO("--------------------------");
        cerr << endl;

        //Extract rgb and depth image from ROS messages
        cv_bridge::CvImageConstPtr rgb_cv_ptr,depth_cv_ptr;
        try{
            rgb_cv_ptr = cv_bridge::toCvShare(rgb_image_msg);
            depth_cv_ptr = cv_bridge::toCvShare(depth_image_msg);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat rgb_image = rgb_cv_ptr->image.clone();
        cv::Mat depth_image;
        depth_cv_ptr->image.convertTo(depth_image,CV_16UC1,1000);
        this->setImages(rgb_image.clone(),depth_image.clone());

        //Listen to depth camera pose
        tf::StampedTransform depth_camera_pose;
        try {
            _listener.waitForTransform("map",
                                       "camera_depth_optical_frame",
                                       ros::Time(0),
                                       ros::Duration(3));
            _listener.lookupTransform("map",
                                      "camera_depth_optical_frame",
                                      ros::Time(0),
                                      depth_camera_pose);
        }
        catch(tf::TransformException ex) {
            ROS_ERROR("%s", ex.what());
        }
        _depth_camera_transform = tfTransform2eigen(depth_camera_pose);
        _inverse_depth_camera_transform = _depth_camera_transform.inverse();

        //detect objects
        this->setDepthCloud(depth_cloud_msg);
        this->detectObjects(logical_image_msg);
        sensor_msgs::ImagePtr label_image_msg = cv_bridge::CvImage(std_msgs::Header(),
                                                                   "bgr8",
                                                                   this->rgbImage()).toImageMsg();
        _label_image_pub.publish(label_image_msg);

        //extract bounding boxes
        Objects objects = extractBoundingBoxes(depth_image);

        //visualize objects
        if(!objects.empty() && _markers_pub.getNumSubscribers() > 0){
            visualization_msgs::MarkerArray markers;

            for(int i=0; i < objects.size(); ++i){
                visualization_msgs::Marker marker;
                const Object& object = objects[i];
                makeMarkerFromObject(marker,object);
                markers.markers.push_back(marker);
            }
            _markers_pub.publish(markers);
        }

        _logical_image_sub.unsubscribe();
        _depth_cloud_sub.unsubscribe();
        _rgb_image_sub.unsubscribe();
        _depth_image_sub.unsubscribe();
    }
}
}
