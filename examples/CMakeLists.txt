if(NOT EXISTS "${STB_INCLUDE_DIR}/stb_image_write.h")
    message(FATAL_ERROR "STB_INCLUDE_DIR must contain stb_image_write.h for examples")
endif()

# Keep executables beside the loader DLL and preserve existing run commands.
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()

# Compile HIP kernels to code objects.
hip_add_executable(
    TARGET render_kernel
    SOURCES simple_render_kernel.hip
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
    INCLUDES ${PROJECT_SOURCE_DIR}/include
)

hip_add_executable(
    TARGET tiled_render_kernel
    SOURCES tiled_render_kernel.hip
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
    INCLUDES ${PROJECT_SOURCE_DIR}/include
)

hip_add_executable(
    TARGET mega_texture_kernel
    SOURCES mega_texture_kernel.hip
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
    INCLUDES ${PROJECT_SOURCE_DIR}/include
)

hip_add_executable(
    TARGET atlas_render_kernel
    SOURCES atlas_render_kernel.hip
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
    INCLUDES ${PROJECT_SOURCE_DIR}/include
)

hip_add_executable(
    TARGET streaming_scene_kernel
    SOURCES streaming_scene_kernel.hip
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
    INCLUDES ${PROJECT_SOURCE_DIR}/include
)

hip_add_executable(
    TARGET path_tracing_kernel
    SOURCES path_tracing_kernel.hip
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
    INCLUDES ${PROJECT_SOURCE_DIR}/include
)

# Example applications (host code only).
add_executable(texture_loader_example simple_render.cpp)
add_executable(tiled_render_example tiled_render.cpp)
add_executable(mega_texture_example mega_texture_render.cpp)
add_executable(atlas_render_example atlas_render.cpp)
add_executable(async_vs_sync_example async_vs_sync.cpp)
add_executable(streaming_scene_example streaming_scene.cpp)
add_executable(path_tracing_example path_tracing.cpp)

target_include_directories(texture_loader_example
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${STB_INCLUDE_DIR}
)

target_include_directories(tiled_render_example
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${STB_INCLUDE_DIR}
)

target_include_directories(mega_texture_example
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${STB_INCLUDE_DIR}
)

target_include_directories(atlas_render_example
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${STB_INCLUDE_DIR}
)

target_include_directories(async_vs_sync_example
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${STB_INCLUDE_DIR}
)

target_include_directories(streaming_scene_example
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${STB_INCLUDE_DIR}
)

target_include_directories(path_tracing_example
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${STB_INCLUDE_DIR}
)

target_link_libraries(texture_loader_example PRIVATE hip_demand_texture)
target_compile_definitions(texture_loader_example PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(tiled_render_example PRIVATE hip_demand_texture)
target_compile_definitions(tiled_render_example PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(mega_texture_example PRIVATE hip_demand_texture)
target_compile_definitions(mega_texture_example PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(atlas_render_example PRIVATE hip_demand_texture)
target_compile_definitions(atlas_render_example PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(async_vs_sync_example PRIVATE hip_demand_texture)
target_compile_definitions(async_vs_sync_example PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(streaming_scene_example PRIVATE hip_demand_texture)
target_compile_definitions(streaming_scene_example PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(path_tracing_example PRIVATE hip_demand_texture)
target_compile_definitions(path_tracing_example PRIVATE __HIP_PLATFORM_AMD__)

# Copy kernel code objects to the executable directory.
add_custom_command(TARGET texture_loader_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:render_kernel,HIP_CODE_OBJECT>
        $<TARGET_FILE_DIR:texture_loader_example>/render_kernel.co
    COMMENT "Copying HIP kernel code object"
)

add_custom_command(TARGET tiled_render_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:tiled_render_kernel,HIP_CODE_OBJECT>
        $<TARGET_FILE_DIR:tiled_render_example>/tiled_render_kernel.co
    COMMENT "Copying HIP kernel code object"
)

add_custom_command(TARGET mega_texture_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:mega_texture_kernel,HIP_CODE_OBJECT>
        $<TARGET_FILE_DIR:mega_texture_example>/mega_texture_kernel.co
    COMMENT "Copying HIP kernel code object"
)

add_custom_command(TARGET atlas_render_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:atlas_render_kernel,HIP_CODE_OBJECT>
        $<TARGET_FILE_DIR:atlas_render_example>/atlas_render_kernel.co
    COMMENT "Copying HIP kernel code object"
)

add_custom_command(TARGET async_vs_sync_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:render_kernel,HIP_CODE_OBJECT>
        $<TARGET_FILE_DIR:async_vs_sync_example>/render_kernel.co
    COMMENT "Copying HIP kernel code object"
)

add_custom_command(TARGET streaming_scene_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:streaming_scene_kernel,HIP_CODE_OBJECT>
        $<TARGET_FILE_DIR:streaming_scene_example>/streaming_scene_kernel.co
    COMMENT "Copying HIP kernel code object"
)

add_custom_command(TARGET path_tracing_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:path_tracing_kernel,HIP_CODE_OBJECT>
        $<TARGET_FILE_DIR:path_tracing_example>/path_tracing_kernel.co
    COMMENT "Copying HIP kernel code object"
)

add_dependencies(texture_loader_example render_kernel)
add_dependencies(tiled_render_example tiled_render_kernel)
add_dependencies(mega_texture_example mega_texture_kernel)
add_dependencies(atlas_render_example atlas_render_kernel)
add_dependencies(async_vs_sync_example render_kernel)
add_dependencies(streaming_scene_example streaming_scene_kernel)
add_dependencies(path_tracing_example path_tracing_kernel)

# Benchmark suite.
hip_add_executable(
    TARGET benchmark_kernel
    SOURCES benchmark_kernel.hip
    ARCHITECTURES ${HIP_ARCHITECTURES}
    OPTIONS -O3 --std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
    INCLUDES ${PROJECT_SOURCE_DIR}/include
)

add_executable(benchmark_suite benchmark_suite.cpp)
target_include_directories(benchmark_suite PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(benchmark_suite PRIVATE hip_demand_texture hip::host)
target_compile_definitions(benchmark_suite PRIVATE __HIP_PLATFORM_AMD__)

add_custom_command(TARGET benchmark_suite POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:benchmark_kernel,HIP_CODE_OBJECT>
        $<TARGET_FILE_DIR:benchmark_suite>/benchmark_kernel.co
    COMMENT "Copying HIP kernel code object"
)

add_dependencies(benchmark_suite benchmark_kernel)

install(TARGETS
    texture_loader_example
    tiled_render_example
    mega_texture_example
    atlas_render_example
    async_vs_sync_example
    streaming_scene_example
    path_tracing_example
    benchmark_suite
    RUNTIME DESTINATION bin
)

install(FILES
    $<TARGET_PROPERTY:render_kernel,HIP_CODE_OBJECT>
    DESTINATION bin
    RENAME render_kernel.co
)

install(FILES
    $<TARGET_PROPERTY:tiled_render_kernel,HIP_CODE_OBJECT>
    DESTINATION bin
    RENAME tiled_render_kernel.co
)

install(FILES
    $<TARGET_PROPERTY:mega_texture_kernel,HIP_CODE_OBJECT>
    DESTINATION bin
    RENAME mega_texture_kernel.co
)

install(FILES
    $<TARGET_PROPERTY:atlas_render_kernel,HIP_CODE_OBJECT>
    DESTINATION bin
    RENAME atlas_render_kernel.co
)

install(FILES
    $<TARGET_PROPERTY:streaming_scene_kernel,HIP_CODE_OBJECT>
    DESTINATION bin
    RENAME streaming_scene_kernel.co
)

install(FILES
    $<TARGET_PROPERTY:path_tracing_kernel,HIP_CODE_OBJECT>
    DESTINATION bin
    RENAME path_tracing_kernel.co
)

install(FILES
    $<TARGET_PROPERTY:benchmark_kernel,HIP_CODE_OBJECT>
    DESTINATION bin
    RENAME benchmark_kernel.co
)

if(USE_OIIO)
    add_executable(oiio_formats_example oiio_formats.cpp)
    target_include_directories(oiio_formats_example PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(oiio_formats_example PRIVATE hip_demand_texture)
    target_compile_definitions(oiio_formats_example PRIVATE __HIP_PLATFORM_AMD__ USE_OIIO)
    install(TARGETS oiio_formats_example RUNTIME DESTINATION bin)
    message(STATUS "Building OpenImageIO format example")
endif()
