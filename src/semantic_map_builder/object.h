#pragma once
#include <string>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace semantic_map_builder {

typedef pcl::PointCloud<Eigen::Vector3f> PointCloudType;

class Object{
public:
    enum Build {MinMax, Centroid};

    Object(const int id_ = 0,
           const std::string &type_ = "",
           const Eigen::Vector3f &a_ = Eigen::Vector3f::Zero(),
           const Eigen::Vector3f &b_ = Eigen::Vector3f::Zero(),
           const Build &b = Build::MinMax);

    bool operator < (const Object &o);
    bool operator == (const Object &o);

    inline const int id() const {return _id;}
    inline const std::string& type() const {return _type;}
    inline const Eigen::Vector3f& lower() const {return _lower;}
    inline const Eigen::Vector3f& uppper() const {return _upper;}
    inline const Eigen::Vector3f& centroid() const {return _centroid;}
    inline const Eigen::Vector3f& halfSize() const {return _half_size;}

protected:
    int _id;
    std::string _type;
    Eigen::Vector3f _lower;
    Eigen::Vector3f _upper;
    Eigen::Vector3f _centroid;
    Eigen::Vector3f _half_size;
    PointCloudType _cloud;
};

typedef std::vector<Object> SemanticMap;

typedef std::pair<Object,Object> Association;

}
