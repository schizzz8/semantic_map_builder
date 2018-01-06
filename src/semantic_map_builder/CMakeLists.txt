add_library(semantic_map_builder_library SHARED
  semantic_map_builder.h semantic_map_builder.cpp
  semantic_map_builder_node.h semantic_map_builder_node.cpp
)
target_link_libraries(semantic_map_builder_library
  ${catkin_LIBRARIES}
)
