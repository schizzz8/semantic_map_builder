#include "object.h"

namespace semantic_map_builder{

using namespace std;

Object::Object(const int id_,
               const string &type_,
               const Eigen::Vector3f &a_,
               const Eigen::Vector3f &b_,
               const Build &b,
               const PointCloudType &cloud_){
    _id = id_;
    _type = type_;

    switch (b) {
    case Build::MinMax:
        _lower = a_;
        _upper = b_;
        _centroid = (a_+b_)/2.0f;
        _half_size = (b_-a_)/2.0f;
        break;
    case Build::Centroid:
        _centroid = a_;
        _half_size = b_;
        _lower = a_-b_;
        _upper = a_+b_;
        break;
    default:
        break;
    }

    _cloud = cloud_;
}

bool Object::operator <(const Object &o){
    return (this->_id < o.id());
}

bool Object::operator ==(const Object &o){
    return (this->_id == o.id());
}

void Object::merge(const Object &o){
    if(o._lower.x() < _lower.x())
        _lower.x() = o._lower.x();
    if(o._upper.x() > _upper.x())
        _upper.x() = o._upper.x();
    if(o._lower.y() < _lower.y())
        _lower.y() = o._lower.y();
    if(o._upper.y() > _upper.y())
        _upper.y() = o._upper.y();
    if(o._lower.z() < _lower.z())
        _lower.z() = o._lower.z();
    if(o._upper.z() > _upper.z())
        _upper.z() = o._upper.z();

    _centroid = (_lower+_upper)/2.0f;
    _half_size = (_lower-_upper)/2.0f;

}
}
