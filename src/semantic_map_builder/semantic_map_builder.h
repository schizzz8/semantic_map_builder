#pragma once

#include <Eigen/Geometry>

#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>

#include "tf/tf.h"
#include "tf/transform_listener.h"
#include "tf/transform_datatypes.h"

#include <lucrezio_logical_camera/LogicalImage.h>

namespace semantic_map_builder {

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
typedef std::vector<Object> Objects;

class SemanticMapBuilder{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SemanticMapBuilder();

    inline void setK(const Eigen::Matrix3f& K_){_K = K_; _invK = _K.inverse();}
    inline void setRGBImage(cv::Mat* rgb_image_){_rgb_image = rgb_image_;}
    inline void setDepthCloud(const PointCloudType::ConstPtr& depth_cloud_){_depth_cloud = depth_cloud_;}

    Detections detectObjects(cv::Mat rgb_image,
                             const Eigen::Isometry3f& depth_camera_transform,
                             const lucrezio_logical_camera::LogicalImage::ConstPtr& logical_image_msg,
                             const PointCloudType::ConstPtr& depth_cloud_msg);

    Objects extractBoundingBoxes(const Detections& detections,
                                 const cv::Mat &depth_image,
                                 const Eigen::Isometry3f& depth_camera_transform);

protected:
    PointCloudType::Ptr transformCloud(const PointCloudType::ConstPtr &in_cloud,
                                       const Eigen::Isometry3f& transform,
                                       const std::string& frame_id);

    PointCloudType::Ptr modelBoundingBox(const lucrezio_logical_camera::Model& model,
                                         const Eigen::Isometry3f& transform);

    PointCloudType::Ptr filterCloud(const PointCloudType::ConstPtr &in_cloud,
                                    const pcl::PointXYZ& min_pt,
                                    const pcl::PointXYZ& max_pt);


    float _raw_depth_scale;
    float _camera_height;
    Eigen::Matrix3f _K,_invK;
    cv::Mat* _rgb_image;
    PointCloudType::ConstPtr _depth_cloud;
    Eigen::Isometry3f tfTransform2eigen(const tf::Transform& p);
    tf::Transform eigen2tfTransform(const Eigen::Isometry3f& T);

};

}
