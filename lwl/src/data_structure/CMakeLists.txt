# some comments take a morning to figure out cross-linking
add_library(data_structure STATIC
    # cuda_matrix.cu this needs to be included directly on cu, template class
    camera_grid.cu
    camera.cu
    camera.cpp
    # some utils
)

target_link_libraries(data_structure
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

target_compile_features(data_structure PUBLIC)
set_target_properties(data_structure PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

pybind11_add_module(pyactivegrid pybind/active_viewpoints_generator_pybind.cpp)

target_link_libraries(pyactivegrid PUBLIC
  data_structure
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

install(TARGETS pyactivegrid DESTINATION lwl)