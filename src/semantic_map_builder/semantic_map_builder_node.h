#pragma once

#include "semantic_map_builder.h"
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <lucrezio_logical_camera/LogicalImage.h>

namespace semantic_map_builder{

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudType;

class SemanticMapBuilderNode : public SemanticMapBuilder{
public:
    SemanticMapBuilderNode(ros::NodeHandle nh_);

    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg);

    void filterCallback(const lucrezio_logical_camera::LogicalImage::ConstPtr& logical_image_msg,
                        const PointCloudType::ConstPtr& depth_cloud_msg,
                        const sensor_msgs::Image::ConstPtr& rgb_image_msg,
                        const sensor_msgs::Image::ConstPtr& depth_image_msg);
protected:
    ros::NodeHandle _nh;

    ros::Subscriber _camera_info_sub;
    bool _got_info;

    tf::TransformListener _listener;

    message_filters::Subscriber<lucrezio_logical_camera::LogicalImage> _logical_image_sub;
    message_filters::Subscriber<PointCloudType> _depth_cloud_sub;
    message_filters::Subscriber<sensor_msgs::Image> _rgb_image_sub;
    message_filters::Subscriber<sensor_msgs::Image> _depth_image_sub;

    typedef message_filters::sync_policies::ApproximateTime<lucrezio_logical_camera::LogicalImage,
                                                            PointCloudType,
                                                            sensor_msgs::Image,
                                                            sensor_msgs::Image> FilterSyncPolicy;
    message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;

};
}
