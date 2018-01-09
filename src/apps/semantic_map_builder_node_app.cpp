#include <iostream>
#include <ros/ros.h>

#include "semantic_map_builder/semantic_map_builder_node.h"

using namespace std;
using namespace semantic_map_builder;

int main(int argc, char** argv){
    ros::init(argc, argv, "semantic_map_builder");

    ros::NodeHandle nh;
    ROS_INFO("Starting semantic_map_builder_node!");
    SemanticMapBuilderNode builder(nh);

    ros::spin();

    ROS_INFO("Done!");
    return 0;
}
