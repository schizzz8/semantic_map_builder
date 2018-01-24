#include <iostream>
#include <ros/ros.h>
#include "semantic_map_builder/builder.h"
#include <sensor_msgs/CameraInfo.h>
#include "tf/tf.h"
#include "tf/transform_listener.h"
#include "tf/transform_datatypes.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "semantic_map_builder/ImageBoundingBoxesArray.h"
#include <visualization_msgs/MarkerArray.h>

using namespace std;
using namespace semantic_map_builder;

constexpr unsigned int str2int(const char* str, int h = 0){
    return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}


class SemanticMapBuilderNode : public Builder{
public:
    SemanticMapBuilderNode(ros::NodeHandle nh_):
        Builder(),
        _nh(nh_),
        _image_bounding_boxes_sub(_nh,"/image_bounding_boxes",100),
        _depth_image_sub(_nh,"/camera/depth/image_raw", 100),
        _synchronizer(FilterSyncPolicy(100),_image_bounding_boxes_sub,_depth_image_sub){

        _raw_depth_scale = 0.001;
        _got_info = false;
        _camera_info_sub = _nh.subscribe("/camera/depth/camera_info",
                                         1000,
                                         &SemanticMapBuilderNode::cameraInfoCallback,
                                         this);

        _synchronizer.registerCallback(boost::bind(&SemanticMapBuilderNode::filterCallback, this, _1, _2));

        _shape = visualization_msgs::Marker::CUBE;
        _markers_pub = _nh.advertise<visualization_msgs::MarkerArray>("visualization_markers",1);
    }


    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg){
        sensor_msgs::CameraInfo camerainfo;
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

    void filterCallback(const semantic_map_builder::ImageBoundingBoxesArray::ConstPtr& image_bounding_boxes_msg,
                        const sensor_msgs::Image::ConstPtr& depth_image_msg){
        if(_got_info && !image_bounding_boxes_msg->image_bounding_boxes.empty()){

            //Extract depth image from ROS message
            cv_bridge::CvImageConstPtr depth_cv_ptr;
            try{
                depth_cv_ptr = cv_bridge::toCvShare(depth_image_msg);
            } catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
            depth_cv_ptr->image.convertTo(_depth_image,CV_16UC1,1000);

            //Listen to camera pose
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
            _globalT = tfTransform2eigen(depth_camera_pose);


            //extract bounding boxes
            Detections detections = imageBoundingBoxes2Detections(image_bounding_boxes_msg);
            extractObjects(detections);

            //data association
            findAssociations();

            //merge maps
            mergeMaps();

            //visualize objects
            if(!_global_map.empty() && _markers_pub.getNumSubscribers() > 0){
                visualization_msgs::MarkerArray markers;

                for(int i=0; i < _global_map.size(); ++i){
                    visualization_msgs::Marker marker;
                    const Object& object = _global_map[i];
                    makeMarkerFromObject(marker,object);
                    markers.markers.push_back(marker);
                }
                _markers_pub.publish(markers);
            }

        }
        std::cerr << ".";
        _image_bounding_boxes_sub.unsubscribe();
        _depth_image_sub.unsubscribe();
    }

protected:
    ros::NodeHandle _nh;

    ros::Subscriber _camera_info_sub;
    bool _got_info;

    tf::TransformListener _listener;

    message_filters::Subscriber<semantic_map_builder::ImageBoundingBoxesArray> _image_bounding_boxes_sub;
    message_filters::Subscriber<sensor_msgs::Image> _depth_image_sub;
    typedef message_filters::sync_policies::ApproximateTime<semantic_map_builder::ImageBoundingBoxesArray,sensor_msgs::Image> FilterSyncPolicy;
    message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;

    semantic_map_builder::Detections _detections;

    uint32_t _shape;
    ros::Publisher _markers_pub;

private:
    Eigen::Isometry3f tfTransform2eigen(const tf::Transform& p){
        Eigen::Isometry3f iso;
        iso.translation().x()=p.getOrigin().x();
        iso.translation().y()=p.getOrigin().y();
        iso.translation().z()=p.getOrigin().z();
        Eigen::Quaternionf q;
        tf::Quaternion tq = p.getRotation();
        q.x()= tq.x();
        q.y()= tq.y();
        q.z()= tq.z();
        q.w()= tq.w();
        iso.linear()=q.toRotationMatrix();
        return iso;
    }

    tf::Transform eigen2tfTransform(const Eigen::Isometry3f& T){
        Eigen::Quaternionf q(T.linear());
        Eigen::Vector3f t=T.translation();
        tf::Transform tft;
        tft.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));
        tft.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
        return tft;
    }

    int type2id(const std::string &type){
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
    void makeMarkerFromObject(visualization_msgs::Marker &marker, const Object &object){
        marker.header.frame_id = "/map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "basic_shapes";
        marker.id = type2id(object.type());
        marker.type = _shape;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = object.centroid().x();
        marker.pose.position.y = object.centroid().y();
        marker.pose.position.z = object.centroid().z();
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = object.halfSize().x();
        marker.scale.y = object.halfSize().y();
        marker.scale.z = object.halfSize().z();

        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0;

        marker.lifetime = ros::Duration();
    }
    semantic_map_builder::Detections imageBoundingBoxes2Detections(const ImageBoundingBoxesArray::ConstPtr &image_bounding_boxes_msg){
        const std::vector<ImageBoundingBox>& image_bounding_boxes = image_bounding_boxes_msg->image_bounding_boxes;
        semantic_map_builder::Detections detections;
        Detection detection;
        std::cerr << "Got " << image_bounding_boxes.size() << " image bounding boxes" << std::endl;
        for(int i=0; i < image_bounding_boxes.size(); ++i){
            std::cerr << "#" << i+1 << std::endl;
            const ImageBoundingBox& image_bounding_box = image_bounding_boxes[i];
            detection._type = image_bounding_box.type;
            std::cerr << "\t>>type: " << detection._type << std::endl;
            detection._top_left = Eigen::Vector2i(image_bounding_box.top_left.r,image_bounding_box.top_left.c);
            detection._bottom_right = Eigen::Vector2i(image_bounding_box.bottom_right.r,image_bounding_box.bottom_right.c);
            std::cerr << "\t>>BB: [(" << detection._top_left.transpose() << ") - (" << detection._bottom_right.transpose() << ")]" << std::endl;
            const std::vector<Pixel> pixels = image_bounding_box.pixels;
            std::cerr << "\t>>Pixels: " << pixels.size() << std::endl;
            for(int j=0; j < pixels.size(); ++j){
                detection._pixels.push_back(Eigen::Vector2i(pixels[j].r,pixels[j].c));
            }
            detections.push_back(detection);
        }
        return detections;
    }
};

int main(int argc, char** argv){
    ros::init(argc, argv, "semantic_map_builder");

    ros::NodeHandle nh;
    ROS_INFO("Starting semantic_map_builder_node!");
    SemanticMapBuilderNode builder(nh);

    ros::spin();

    ROS_INFO("Done!");
    return 0;
}
