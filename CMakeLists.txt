cmake_minimum_required(VERSION 3.20)
project(akdlaa CXX)

set(CMAKE_CXX_STANDARD 17)
set(RESHADE_INC "" CACHE PATH "ReShade include/ directory (contains reshade.hpp)")
set(NGX_ROOT "" CACHE PATH "NVIDIA DLSS SDK root (contains sdk/include and sdk/lib)")

add_library(dlaa SHARED dlaa_addon.cpp)
set_target_properties(dlaa PROPERTIES SUFFIX ".addon64" PREFIX "")
target_include_directories(dlaa PRIVATE ${RESHADE_INC} ${NGX_ROOT}/include)
target_link_directories(dlaa PRIVATE ${NGX_ROOT}/lib/Windows_x86_64/x64)
target_link_libraries(dlaa PRIVATE nvsdk_ngx_d d3d11 dxgi d3dcompiler)
