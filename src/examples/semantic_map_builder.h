#pragma once

#include <Eigen/Geometry>

#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "tf/tf.h"
#include "tf/transform_listener.h"
#include "tf/transform_datatypes.h"

#include <lucrezio_logical_camera/LogicalImage.h>

namespace semantic_map_builder {

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudType;

class Detection{
public:
    Detection(const std::string& type_="",
              const Eigen::Vector2i& p_min_ = Eigen::Vector2i(10000,10000),
              const Eigen::Vector2i& p_max_ = Eigen::Vector2i(-10000,-10000),
              const std::vector<Eigen::Vector2i>& pixels_ = std::vector<Eigen::Vector2i>()):
        type(type_),
        p_min(p_min_),
        p_max(p_max_),
        pixels(pixels_){}
    std::string type;
    Eigen::Vector2i p_min;
    Eigen::Vector2i p_max;
    std::vector<Eigen::Vector2i> pixels;
};

class Object{
public:
    Object(const std::string& type_ = "",
           const Eigen::Vector3f& centroid_ = Eigen::Vector3f::Zero(),
           const Eigen::Vector3f& size_ = Eigen::Vector3f::Zero()):
        type(type_),centroid(centroid_),size(size_){}
    std::string type;
    Eigen::Vector3f centroid;
    Eigen::Vector3f size;
};

typedef std::vector<Detection> Detections;
typedef std::vector<Object> Objects;
typedef std::vector<lucrezio_logical_camera::Model> Models;
typedef std::pair<Eigen::Vector3f,Eigen::Vector3f> BoundingBox3D;
typedef std::vector<BoundingBox3D> BoundingBoxes3D;

class SemanticMapBuilder{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SemanticMapBuilder();

    inline void setK(const Eigen::Matrix3f& K_){_K = K_; _invK = _K.inverse();}
    void setImages(const cv::Mat& rgb_image_, const cv::Mat& depth_image_);
    inline void setDepthCloud(const PointCloudType::ConstPtr& depth_cloud_){_depth_cloud = depth_cloud_;}
    inline const cv::Mat& rgbImage(){return _rgb_image;}

    void detectObjects(const lucrezio_logical_camera::LogicalImage::ConstPtr& logical_image_msg);
    Objects extractBoundingBoxes(const cv::Mat &depth_image);

protected:

    void computeWorldBoundingBoxes(BoundingBoxes3D &bounding_boxes,const Eigen::Isometry3f &transform,const Models &models);
    void computeImageBoundingBoxes(const BoundingBoxes3D &bounding_boxes);

    inline bool inRange(const pcl::PointXYZ &point, const BoundingBox3D &bounding_box){
        return (point.x >= bounding_box.first.x() && point.x < bounding_box.second.x() &&
                point.y >= bounding_box.first.y() && point.y < bounding_box.second.y() &&
                point.z >= bounding_box.first.z() && point.z < bounding_box.second.z());
    }

    float _raw_depth_scale;
    float _camera_height;
    float _focal_length;
    Eigen::Matrix3f _K,_invK;
    Eigen::Isometry3f _depth_camera_transform;
    Eigen::Isometry3f _inverse_depth_camera_transform;
    cv::Mat _rgb_image;
    cv::Mat _depth_image;
    PointCloudType::ConstPtr _depth_cloud;
    Detections _detections;

    Eigen::Isometry3f tfTransform2eigen(const tf::Transform& p);
    tf::Transform eigen2tfTransform(const Eigen::Isometry3f& T);
};

}
