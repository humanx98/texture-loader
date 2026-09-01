#!/bin/bash

set -e

script_dir=$(dirname $0)
root_dir=$(realpath "$script_dir/../..")

cmake -S "$root_dir" -B ./build \
    -G "Visual Studio 18 2026" \
    -A x64 \
    -T "version=14.50" \
    -DBUILD_EXAMPLES=ON \
    -DCMAKE_TOOLCHAIN_FILE=C:/my_space/code/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DUSE_OIIO=ON \
    -DHIP_ARCHITECTURES=gfx1031