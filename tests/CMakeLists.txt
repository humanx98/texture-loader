if(USE_VCPKG)
    find_package(GTest CONFIG REQUIRED)
else()
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# Keep executables beside the loader DLL and preserve existing run commands.
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()

add_test(NAME vcpkg_bootstrap_config
    COMMAND ${CMAKE_COMMAND}
        "-DTEXTURE_LOADER_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/VcpkgBootstrapTests.cmake"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(texture_loader_tests
    main.cpp
    TextureInfoTests.cpp
    DemandTextureLoaderTests.cpp
    TextureUploadTests.cpp
    ThreadPoolTests.cpp
    MemoryPoolTests.cpp
    TicketTests.cpp
)

target_include_directories(texture_loader_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${STB_INCLUDE_DIR}
)

target_link_libraries(texture_loader_tests
    PRIVATE
        hip_demand_texture
        GTest::gtest
        GTest::gtest_main
)

target_compile_definitions(texture_loader_tests PRIVATE __HIP_PLATFORM_AMD__)

if(USE_OIIO)
    target_compile_definitions(texture_loader_tests PRIVATE USE_OIIO)
    target_sources(texture_loader_tests PRIVATE OIIOReaderTests.cpp)
    target_link_libraries(texture_loader_tests PRIVATE OpenImageIO::OpenImageIO)
endif()

include(GoogleTest)
gtest_discover_tests(texture_loader_tests WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

# Native image decode regressions do not initialize a GPU. This separate
# target can also be built and run with address/undefined sanitizers.
add_executable(image_data_tests ImageDataTests.cpp)
target_include_directories(image_data_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(image_data_tests PRIVATE hip::host GTest::gtest_main)
target_compile_definitions(image_data_tests PRIVATE __HIP_PLATFORM_AMD__)
if(WIN32)
    add_dependencies(image_data_tests hip_runtime_dlls)
endif()
gtest_discover_tests(image_data_tests WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

message(STATUS "Building unit tests")
