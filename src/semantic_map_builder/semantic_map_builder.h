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
    Detection(const std::string& type_=0,
              const Eigen::Vector2i& p_min_ = Eigen::Vector2i::Zero(),
              const Eigen::Vector2i& p_max_ = Eigen::Vector2i::Zero(),
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
                             const lucrezio_logical_camera::LogicalImage::ConstPtr& logical_image_msg,
                             const PointCloudType::ConstPtr& depth_cloud_msg);

    Objects extractBoundingBoxes(const Detections& detections,
                                 const cv::Mat &depth_image);

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
    float _focal_length;
    Eigen::Matrix3f _K,_invK;
    Eigen::Isometry3f _depth_camera_transform;
    cv::Mat* _rgb_image;
    PointCloudType::ConstPtr _depth_cloud;
    Eigen::Isometry3f tfTransform2eigen(const tf::Transform& p);
    tf::Transform eigen2tfTransform(const Eigen::Isometry3f& T);
};

}
