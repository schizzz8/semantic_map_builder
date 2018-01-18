add_library(semantic_map_builder_library SHARED
  object.h object.cpp
  detection.h detection.cpp
  builder.h builder.cpp
)
target_link_libraries(semantic_map_builder_library
  ${catkin_LIBRARIES}
)

