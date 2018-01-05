#include <iostream>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <lucrezio_logical_camera/LogicalImage.h>

#include "tf/tf.h"
#include "tf/transform_listener.h"
#include "tf/transform_datatypes.h"

#include <Eigen/Geometry>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>


using namespace std;
using namespace message_filters;
using namespace lucrezio_logical_camera;
using namespace sensor_msgs;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudType;

class Detection{
public:
    Detection(std::string type_=0,
              float x_ = 0.0f,
              float y_ = 0.0f,
              float width_ = 0.0f,
              float height_ = 0.0f):
        _type(type_),
        _x(x_),
        _y(y_),
        _width(width_),
        _height(height_){}

    inline const std::string& type() const {return _type;}
    inline const float& x() const {return _x;}
    inline const float& y() const {return _y;}
    inline const float& width() const {return _width;}
    inline const float& height() const {return _height;}
private:
    std::string _type;
    float _x;
    float _y;
    float _width;
    float _height;
};

class Object{
public:
    Object(const std::string& type_ = "",
           const Eigen::Vector3f& centroid_ = Eigen::Vector3f::Zero(),
           const Eigen::Vector3f& size_ = Eigen::Vector3f::Zero()):
        _type(type_),_centroid(centroid_), _size(size_){}
private:
    std::string _type;
    Eigen::Vector3f _centroid;
    Eigen::Vector3f _size;
};

typedef std::vector<Detection> Detections;

class SemanticMapBuilder{
public:
    SemanticMapBuilder(std::string robotname_ = ""):
        _robotname(robotname_),
        _logical_image_sub(_nh,"/gazebo/logical_camera_image",1),
        _depth_cloud_sub(_nh,"/camera/depth/points",1),
        _rgb_image_sub(_nh,"/camera/rgb/image_raw", 1),
        _depth_image_sub(_nh,"/camera/depth/image_raw", 1),
        _synchronizer(FilterSyncPolicy(10),_logical_image_sub,_depth_cloud_sub,_rgb_image_sub,_depth_image_sub){

        _raw_depth_scale = 0.001;
        _camera_height = 1.0f;
        _got_info = false;
        _camera_info_sub = _nh.subscribe("/camera/depth/camera_info",
                                         1000,
                                         &SemanticMapBuilder::cameraInfoCallback,
                                         this);

        _synchronizer.registerCallback(boost::bind(&SemanticMapBuilder::filterCallback, this, _1, _2, _3, _4));
        ROS_INFO("Starting training set generator node!");
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

    void filterCallback(const LogicalImage::ConstPtr& logical_image_msg,
                        const PointCloudType::ConstPtr& depth_cloud_msg,
                        const Image::ConstPtr& rgb_image_msg,
                        const Image::ConstPtr& depth_image_msg){
        if(_got_info){

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
            Eigen::Isometry3f depth_camera_transform = tfTransform2eigen(depth_camera_pose);

            //Detect Objects
            Detections detections = detectObjects(depth_camera_transform,
                                                  logical_image_msg,
                                                  depth_cloud_msg);

            //Extract Bounding Boxes
            for(int i=0; i < detections.size(); ++i){

                Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
                Eigen::Vector3f size = Eigen::Vector3f::Zero();

                const Detection& detection = detections[i];

                cv::Mat roi = depth_image(cv::Rect(detection.x(),
                                                   detection.y(),
                                                   detection.width(),
                                                   detection.height()));
                int roi_rows = roi.rows;
                int roi_cols = roi.cols;
                cv::Mat roi_without_floor;
                roi_without_floor.create(roi_rows,roi_cols,CV_16UC1);
                roi_without_floor = std::numeric_limits<ushort>::max();

                float min_distance=100;
                int closest_r=0;
                int closest_c=0;
                for(int r = 0; r < roi_rows; ++r){
                    const ushort* roi_ptr  = roi.ptr<ushort>(r);
                    ushort* out_ptr  = roi_without_floor.ptr<ushort>(r);
                    for(int c=0; c < roi_cols; ++c, ++roi_ptr, ++out_ptr) {
                        const ushort& depth = *roi_ptr;
                        ushort& out = *out_ptr;
                        float d = depth * _raw_depth_scale;
                        Eigen::Vector3f point = _invK * Eigen::Vector3f(c*d,r*d,d);

                        if(point.y() < _camera_height){
                            out = depth;
                            float distance = point.squaredNorm();
                            if(distance<min_distance){
                                min_distance = distance;
                                closest_r = r;
                                closest_c = c;
                            }
                        }
                    }
                }

                cv::Mat mask;
                mask.create(roi_rows+2,roi_cols+2,CV_8UC1);
                mask = 0;
                cv::Point closest_point(closest_r,closest_c);
                cv::Rect ccomp;
                cv::floodFill(roi_without_floor,
                              mask,
                              closest_point,
                              255,
                              &ccomp,
                              cv::Scalar(0.025),
                              cv::Scalar(0.025));

                int k=0;
                float x_min=1000000,x_max=-1000000,y_min=1000000,y_max=-1000000,z_min=1000000,z_max=-1000000;
                for(int r = 0; r < roi_rows; ++r){
                    const ushort* roi_ptr  = roi_without_floor.ptr<ushort>(r);
                    for(int c=0; c < roi_cols; ++c, ++roi_ptr) {
                        const ushort& depth = *roi_ptr;
                        float d = depth * _raw_depth_scale;
                        Eigen::Vector3f world_point = depth_camera_transform*_invK * Eigen::Vector3f(c*d,r*d,d);
                        centroid += world_point;

                        if(world_point.x()<x_min)
                            x_min = world_point.x();
                        if(world_point.x()>x_max)
                            x_max = world_point.x();
                        if(world_point.y()<y_min)
                            y_min = world_point.y();
                        if(world_point.y()>y_max)
                            y_max = world_point.y();
                        if(world_point.z()<z_min)
                            z_min = world_point.z();
                        if(world_point.z()>z_max)
                            z_max = world_point.z();

                        k++;
                    }
                }
                centroid = centroid/(float)k;
                size = Eigen::Vector3f(x_max-x_min,y_max-y_min,z_max-z_min);

                _local_map.push_back(Object(detection.type(),
                                            centroid,
                                            size));

            }
        }
    }

    Detections detectObjects(const Eigen::Isometry3f& depth_camera_transform,
                             const lucrezio_logical_camera::LogicalImage::ConstPtr& logical_image_msg,
                             const PointCloudType::ConstPtr& depth_cloud_msg){

        Detections detections;

        PointCloudType::Ptr local_map_cloud (new PointCloudType ());
        pcl::transformPointCloud (*depth_cloud_msg, *local_map_cloud, depth_camera_transform);
        local_map_cloud->header.frame_id = "/map";
        local_map_cloud->width  = depth_cloud_msg->width;
        local_map_cloud->height = depth_cloud_msg->height;
        local_map_cloud->is_dense = false;

        tf::StampedTransform logical_camera_pose;
        tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);
        for(int i=0; i < logical_image_msg->models.size(); i++){

            Eigen::Vector3f box_min (logical_image_msg->models.at(i).min.x,
                                     logical_image_msg->models.at(i).min.y,
                                     logical_image_msg->models.at(i).min.z);

            Eigen::Vector3f box_max (logical_image_msg->models.at(i).max.x,
                                     logical_image_msg->models.at(i).max.y,
                                     logical_image_msg->models.at(i).max.z);

            float x_range = box_max.x()-box_min.x();
            float y_range = box_max.y()-box_min.y();
            float z_range = box_max.z()-box_min.z();

            PointCloudType::Ptr bounding_box_cloud (new PointCloudType ());
            for(int k=0; k <= 1; k++)
                for(int j=0; j <= 1; j++)
                    for(int i=0; i <= 1; i++){
                        bounding_box_cloud->points.push_back (pcl::PointXYZ(box_min.x() + i*x_range,
                                                                            box_min.y() + j*y_range,
                                                                            box_min.z() + k*z_range));
                    }
            PointCloudType::Ptr transformed_bounding_box_cloud (new PointCloudType ());
            tf::Transform model_pose;
            tf::poseMsgToTF(logical_image_msg->models.at(i).pose,model_pose);
            Eigen::Isometry3f model_transform = tfTransform2eigen(logical_camera_pose)*tfTransform2eigen(model_pose);
            pcl::transformPointCloud (*bounding_box_cloud, *transformed_bounding_box_cloud, model_transform);
            pcl::PointXYZ min_pt,max_pt;
            pcl::getMinMax3D(*transformed_bounding_box_cloud,min_pt,max_pt);

            PointCloudType::Ptr cloud_filtered_x (new PointCloudType ());
            PointCloudType::Ptr cloud_filtered_xy (new PointCloudType ());
            PointCloudType::Ptr cloud_filtered_xyz (new PointCloudType ());

            pcl::PassThrough<pcl::PointXYZ> pass;
            pass.setInputCloud (local_map_cloud);
            pass.setFilterFieldName ("x");
            pass.setFilterLimits (min_pt.x,max_pt.x);
            pass.filter (*cloud_filtered_x);

            pass.setInputCloud (cloud_filtered_x);
            pass.setFilterFieldName ("y");
            pass.setFilterLimits (min_pt.y,max_pt.y);
            pass.filter (*cloud_filtered_xy);

            pass.setInputCloud (cloud_filtered_xy);
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (min_pt.z,max_pt.z);
            pass.filter (*cloud_filtered_xyz);

            if(!cloud_filtered_xyz->points.empty()){
                cv::Point2i p_min(10000,10000);
                cv::Point2i p_max(-10000,-10000);

                for(int i=0; i<cloud_filtered_xyz->points.size(); i++){
                    Eigen::Vector3f camera_point = depth_camera_transform.inverse()*
                            Eigen::Vector3f(cloud_filtered_xyz->points[i].x,
                                            cloud_filtered_xyz->points[i].y,
                                            cloud_filtered_xyz->points[i].z);
                    Eigen::Vector3f image_point = _K*camera_point;

                    const float& z=image_point.z();
                    image_point.head<2>()/=z;
                    int r = image_point.x();
                    int c = image_point.y();

                    if(r < p_min.x)
                        p_min.x = r;
                    if(r > p_max.x)
                        p_max.x = r;

                    if(c < p_min.y)
                        p_min.y = c;
                    if(c > p_max.y)
                        p_max.y = c;
                }
                detections.push_back(Detection(logical_image_msg->models.at(i).type,
                                               (float)(p_min.y+p_max.y)/2.0f,
                                               (float)(p_min.x+p_max.x)/2.0f,
                                               (float)(p_max.y-p_min.y),
                                               (float)(p_max.x-p_min.x)));
            }
        }
        return detections;
    }

private:
    ros::NodeHandle _nh;
    string _robotname;

    ros::Subscriber _camera_info_sub;
    Eigen::Matrix3f _K,_invK;
    float _raw_depth_scale;
    float _camera_height;
    bool _got_info;

    tf::TransformListener _listener;

    message_filters::Subscriber<LogicalImage> _logical_image_sub;
    message_filters::Subscriber<PointCloudType> _depth_cloud_sub;
    message_filters::Subscriber<Image> _rgb_image_sub;
    message_filters::Subscriber<Image> _depth_image_sub;
    typedef sync_policies::ApproximateTime<LogicalImage,PointCloudType,Image,Image> FilterSyncPolicy;
    message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;

    int _seq;

    std::vector<Object> _local_map;

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

    const float low=-std::numeric_limits<int>::max();
    const float up=std::numeric_limits<int>::max();

};

int main(int argc, char** argv){
    return 0;
}
