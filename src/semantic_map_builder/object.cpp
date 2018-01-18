#include "object.h"

namespace semantic_map_builder{

using namespace std;

Object::Object(const int id_,
               const string &type_,
               const Eigen::Vector3f &a_,
               const Eigen::Vector3f &b_,
               const Build &b){
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
}

bool Object::operator <(const Object &o){
    return (this->_id < o.id());
}

bool Object::operator ==(const Object &o){
    return (this->_id == o.id());
}

}
