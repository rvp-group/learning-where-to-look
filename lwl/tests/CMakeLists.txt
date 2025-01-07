enable_testing()

    add_executable(test_algebra_operations test_algebra_operations.cu)
    target_link_libraries(test_algebra_operations PRIVATE GTest::gtest_main data_structure)

    add_executable(test_cuda_matrix test_cuda_matrix.cu)
    target_link_libraries(test_cuda_matrix PRIVATE GTest::gtest_main data_structure)

    add_executable(test_projections test_projections.cu)
    target_link_libraries(test_projections PRIVATE GTest::gtest_main data_structure)

    add_executable(test_camera_grid test_camera_grid.cu)
    target_link_libraries(test_camera_grid GTest::gtest_main data_structure)

    include(GoogleTest)
        gtest_discover_tests(test_algebra_operations)
        gtest_discover_tests(test_cuda_matrix)
        gtest_discover_tests(test_projections)
        gtest_discover_tests(test_camera_grid)