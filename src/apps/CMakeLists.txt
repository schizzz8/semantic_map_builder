add_executable(semantic_map_builder_node_app semantic_map_builder_node_app.cpp)
target_link_libraries(semantic_map_builder_node_app
  semantic_map_builder_library
  ${catkin_LIBRARIES}
)

#add_executable(semantic_map_builder_node_example semantic_map_builder_node_example.cpp)
#target_link_libraries(semantic_map_builder_node_example
#  ${catkin_LIBRARIES}
#)
