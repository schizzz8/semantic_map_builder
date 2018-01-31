#include "builder.h"

namespace semantic_map_builder{

using namespace std;

Builder::Builder(){
    _globalT = Eigen::Isometry3f::Identity();

    _min_distance = 0.02;
    _max_distance = 5.0;

    _local_set = false;
    _global_set = false;
}

PointCloudType::Ptr Builder::unproject(const std::vector<Eigen::Vector2i> &pixels){
    PointCloudType::Ptr cloud (new PointCloudType);
    for(int idx=0; idx < pixels.size(); ++idx){
        const Eigen::Vector2i& pixel = pixels[idx];
        int r = pixel.x();
        int c = pixel.y();
        const unsigned short& depth = _depth_image.at<const unsigned short>(r,c);
        float d = depth * _raw_depth_scale;

        if(d <= _min_distance)
            continue;

        if(d >= _max_distance)
            continue;

        Eigen::Vector3f camera_point = _invK * Eigen::Vector3f(c*d,r*d,d);
        Eigen::Vector3f map_point = _globalT*camera_point;
//        std::cerr << map_point.transpose() << " ";
        cloud->push_back(pcl::PointXYZ(map_point.x(),map_point.y(),map_point.z()));

    }
    return cloud;
}

void Builder::getLowerUpper3d(const PointCloudType &cloud, Eigen::Vector3f &lower, Eigen::Vector3f &upper){
    lower.x() = std::numeric_limits<float>::max();
    lower.y() = std::numeric_limits<float>::max();
    lower.z() = std::numeric_limits<float>::max();
    upper.x() = -std::numeric_limits<float>::max();
    upper.y() = -std::numeric_limits<float>::max();
    upper.z() = -std::numeric_limits<float>::max();

    for(int i=0; i < cloud.size(); ++i){

        if(cloud.points[i].x < lower.x())
            lower.x() = cloud.points[i].x;
        if(cloud.points[i].x > upper.x())
            upper.x() = cloud.points[i].x;
        if(cloud.points[i].y < lower.y())
            lower.y() = cloud.points[i].y;
        if(cloud.points[i].y > upper.y())
            upper.y() = cloud.points[i].y;
        if(cloud.points[i].z < lower.z())
            lower.z() = cloud.points[i].z;
        if(cloud.points[i].z > upper.z())
            upper.z() = cloud.points[i].z;
    }
}

void Builder::extractObjects(const Detections &detections){

    bool populate_global = false;
    if(!_global_set){
        populate_global = true;
        _global_set = true;
    } else {
        _local_map.clear();
        _local_set = true;
    }

    for(int i=0; i < detections.size(); ++i){

        const Detection& detection = detections[i];

        cerr << detection._type << ": [(";
        cerr << detection._top_left.transpose() << ") - (" << detection._bottom_right.transpose() << ")]" << endl;

        if((detection._bottom_right-detection._top_left).norm() < 1e-3)
            continue;

        std::cerr << "Unprojecting" << std::endl;
        PointCloudType::Ptr cloud = unproject(detection._pixels);

        PointCloudType::Ptr cloud_filtered (new PointCloudType);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud (cloud);
        sor.setMeanK (10);
        sor.setStddevMulThresh (1.0);
        sor.filter (*cloud_filtered);

        Eigen::Vector3f lower,upper;
        getLowerUpper3d(*cloud_filtered,lower,upper);

        std::string object_type = detection._type.substr(0,detection._type.find_first_of("_"));

        std::cerr << "BB: [(" << lower.transpose() << "," << upper.transpose() << ")]" << std::endl;

        if(populate_global)
            _global_map.push_back(Object(i,
                                         object_type,
                                         lower,
                                         upper,
                                         Object::Build::MinMax,
                                         *cloud_filtered));
        else
            _local_map.push_back(Object(i,
                                        object_type,
                                        lower,
                                        upper,
                                        Object::Build::MinMax,
                                        *cloud_filtered));
    }
}

void Builder::findAssociations(){

    if(!_global_set || !_local_set)
        return;

    const int local_size = _local_map.size();
    const int global_size = _global_map.size();

    std::cerr << "[Data Association] ";
    std::cerr << "{Local Map size: " << local_size << "} ";
    std::cerr << "- {Global Map size: " << global_size << "}" << std::endl;

    _associations.clear();

    for(int i=0; i < global_size; ++i){
        const Object &global = _global_map[i];
        const string &global_type = global.type();

        std::cerr << "\t>> Global: " << global_type;

        Object local_best;
        float best_error = std::numeric_limits<float>::max();

        for(int j=0; j < local_size; ++j){
            const Object &local = _local_map[j];
            const string &local_type = local.type();

            if(local_type != global_type)
                continue;

            Eigen::Vector3f e_c = local.centroid() - global.centroid();

            float error = e_c.transpose()*e_c;

            if(error<best_error){
                best_error = error;
                local_best = local;
            }
        }
        if(local_best.type() == "")
            continue;

        std::cerr << " - Local ID: " << local_best.type() << std::endl;

        _associations.push_back(Association(global,local_best));
    }
}

int Builder::associationID(const Object &local){
    for(int i=0; i < _associations.size(); ++i)
        if(_associations[i].second.id() == local.id())
            return _associations[i].first.id();
    return -1;
}

void Builder::mergeMaps(){
    if(!_global_set || !_local_set)
        return;

    int added = 0, merged = 0;
    for(int i=0; i < _local_map.size(); ++i){
        Object &local = _local_map[i];
        int association_id = associationID(local);
        if(association_id == -1){
            _global_map.push_back(local);
            added++;
            continue;
        } else {
            Object &global_associated = _global_map[association_id];

            if(local.type() != global_associated.type())
                continue;

            global_associated.merge(local);
        }
    }
}
}
