#include "detection.h"

namespace semantic_map_builder {

using namespace std;

Detection::Detection(const string &type_,
                     const Eigen::Vector2i &top_left_,
                     const Eigen::Vector2i &bottom_right_,
                     const std::vector<Eigen::Vector2i> &pixels_):
    _type(type_),
    _top_left(top_left_),
    _bottom_right(bottom_right_),
    _pixels(pixels_){}

}
