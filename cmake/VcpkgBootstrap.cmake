include_guard(GLOBAL)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Enable vcpkg in the parent project before project(), or use USE_VCPKG=OFF here.")
endif()

set(VCPKG_MANIFEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "vcpkg manifest directory")
# Recompute project-owned features on every configure, including ON -> OFF.
list(REMOVE_ITEM VCPKG_MANIFEST_FEATURES oiio tests)
if(USE_OIIO)
    list(APPEND VCPKG_MANIFEST_FEATURES oiio)
endif()
if(BUILD_TESTS)
    list(APPEND VCPKG_MANIFEST_FEATURES tests)
endif()
set(VCPKG_MANIFEST_FEATURES "${VCPKG_MANIFEST_FEATURES}" CACHE STRING "vcpkg manifest features" FORCE)

if(CMAKE_TOOLCHAIN_FILE)
    if(NOT CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg\\.cmake$")
        message(FATAL_ERROR
            "USE_VCPKG requires the vcpkg toolchain. Use VCPKG_CHAINLOAD_TOOLCHAIN_FILE "
            "for another toolchain, or set USE_VCPKG=OFF to manage dependencies yourself.")
    endif()
elseif(DEFINED ENV{VCPKG_ROOT} AND NOT "$ENV{VCPKG_ROOT}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_root)
    if(NOT EXISTS "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
        message(FATAL_ERROR "VCPKG_ROOT does not contain scripts/buildsystems/vcpkg.cmake: ${_vcpkg_root}")
    endif()
    set(CMAKE_TOOLCHAIN_FILE "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain")
else()
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json" _vcpkg_manifest)
    string(JSON _vcpkg_baseline GET "${_vcpkg_manifest}" builtin-baseline)
    # Each pin has its own checkout; never reset a caller's vcpkg installation.
    set(_vcpkg_root "${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/${_vcpkg_baseline}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg")
    file(LOCK "${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/bootstrap.lock" GUARD FILE TIMEOUT 600)
    if(NOT EXISTS "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
        find_package(Git REQUIRED)
        if(NOT EXISTS "${_vcpkg_root}/.git")
            execute_process(COMMAND "${GIT_EXECUTABLE}" init "${_vcpkg_root}"
                COMMAND_ERROR_IS_FATAL ANY)
        endif()
        message(STATUS "Downloading vcpkg at ${_vcpkg_baseline}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${_vcpkg_root}" config remote.origin.url
                https://github.com/microsoft/vcpkg.git
            COMMAND_ERROR_IS_FATAL ANY)
        # Keep registry history available for versioned dependency resolution.
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${_vcpkg_root}" fetch --filter=tree:0 origin "${_vcpkg_baseline}"
            COMMAND_ERROR_IS_FATAL ANY)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${_vcpkg_root}" checkout --detach "${_vcpkg_baseline}"
            COMMAND_ERROR_IS_FATAL ANY)
    endif()
    set(CMAKE_TOOLCHAIN_FILE "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain")
endif()

# vcpkg's toolchain bootstraps its executable and installs the selected manifest.
list(APPEND VCPKG_BOOTSTRAP_OPTIONS "-disableMetrics")
list(REMOVE_DUPLICATES VCPKG_BOOTSTRAP_OPTIONS)
