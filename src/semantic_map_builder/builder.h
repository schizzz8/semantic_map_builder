#pragma once

#include "object.h"
#include "detection.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/filters/statistical_outlier_removal.h>

namespace semantic_map_builder {

class Builder{
public:
    Builder();

    void extractObjects(const Detections &detections);

    void findAssociations();

    void mergeMaps();

protected:
    float _raw_depth_scale;
    float _min_distance, _max_distance;
    Eigen::Matrix3f _K,_invK;
    cv::Mat _depth_image;

    bool _local_set;
    bool _global_set;

    SemanticMap _local_map;
    SemanticMap _global_map;
    Eigen::Isometry3f _globalT;

    std::vector<Association> _associations;

private:
    PointCloudType::Ptr unproject(const std::vector<Eigen::Vector2i> &pixels);
    void getLowerUpper3d(const PointCloudType &cloud, Eigen::Vector3f &lower, Eigen::Vector3f &upper);
    int associationID(const Object &local);
};

}
