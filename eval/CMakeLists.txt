add_executable(run test.cpp)

target_link_libraries(
    run
    PRIVATE llm_system
)

set_target_properties(run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(test_nearbank test_nearbank.cpp)

target_link_libraries(
    test_nearbank
    PRIVATE llm_system
)

set_target_properties(test_nearbank PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
