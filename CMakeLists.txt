CMAKE_MINIMUM_REQUIRED (VERSION 3.14)

PROJECT (llm_system)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

#### Dependency versions ####
set(YAML_CPP_VERSION "0.9.0")
set(FMT_VERSION "12.1.0")
set(SPDLOG_VERSION "1.17.0")
set(ARGPARSE_VERSION "3.2")

#### External libraries ####
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Skip updating the external dependencies after populating them for the first time")

message(STATUS "Configuring yaml-cpp ${YAML_CPP_VERSION}...")
option(YAML_CPP_BUILD_CONTRIB "Enable yaml-cpp contrib in library" OFF)
option(YAML_CPP_BUILD_TOOLS "Enable parse tools" OFF)
option(YAML_BUILD_SHARED_LIBS "Build yaml-cpp as a shared library" OFF)
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        yaml-cpp-${YAML_CPP_VERSION}
)
FetchContent_MakeAvailable(yaml-cpp)
include_directories(${yaml-cpp_SOURCE_DIR}/include)

message(STATUS "Configuring fmt ${FMT_VERSION}...")
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        ${FMT_VERSION}
)
FetchContent_MakeAvailable(fmt)
include_directories(${fmt_SOURCE_DIR}/include)

message(STATUS "Configuring spdlog ${SPDLOG_VERSION}...")
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use external fmt library" FORCE)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v${SPDLOG_VERSION}
)
FetchContent_MakeAvailable(spdlog)
include_directories(${spdlog_SOURCE_DIR}/include)

message(STATUS "Configuring argparse ${ARGPARSE_VERSION}...")
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG        v${ARGPARSE_VERSION}
)
FetchContent_MakeAvailable(argparse)
include_directories(${argparse_SOURCE_DIR}/include)

add_library(llm_system SHARED)
add_compile_options(-fPIC)


include_directories(src)
include_directories(src/dram/ramulator2/src)
add_subdirectory(src)
add_subdirectory(eval)


target_link_libraries(
    llm_system
    PRIVATE hardware
    PRIVATE dram
    PRIVATE module
    PRIVATE model
    PRIVATE scheduler
)

if(APPLE)
    target_link_options(llm_system PRIVATE -Wl,-force_load,$<TARGET_FILE:dram>)
else()
    target_link_options(llm_system PRIVATE -Wl,--whole-archive $<TARGET_FILE:dram> -Wl,--no-whole-archive)
endif()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

configure_file(dram_config_HBM3_80GB.yaml ${EXECUTABLE_OUTPUT_PATH}/dram_config_HBM3_80GB.yaml COPYONLY)
configure_file(dram_config_HBM3E_192GB.yaml ${EXECUTABLE_OUTPUT_PATH}/dram_config_HBM3E_192GB.yaml COPYONLY)
configure_file(config.yaml ${EXECUTABLE_OUTPUT_PATH}/config.yaml COPYONLY)