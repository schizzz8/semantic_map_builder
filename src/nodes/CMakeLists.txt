#add_executable(semantic_map_builder_node semantic_map_builder_node.cpp)
#target_link_libraries(semantic_map_builder_node
#  semantic_map_builder_library
#  ${catkin_LIBRARIES}
#)

add_executable(detection_simulator_node detection_simulator_node.cpp)
target_link_libraries(detection_simulator_node
  ${catkin_LIBRARIES}
)

