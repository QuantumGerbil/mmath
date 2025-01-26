cmake_minimum_required(VERSION 3.28)

project(mmath-test)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_C_COMPILER clang)
#set(CMAKE_CXX_COMPILER clang++)
# Force the final executable to appear in the project root:
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
add_compile_definitions(-D_LIBCPP_REMOVE_TRANSITIVE_INCLUDES)
add_compile_options(-fdiagnostics-color -pedantic-errors -Wall -Wextra -Wdeprecated -fexceptions
#  -fexperimental-library -stdlib=libc++
)
add_link_options(-fuse-ld=lld -Wl,--color-diagnostics -fdiagnostics-color
#  -stdlib=libc++
#  -fexperimental-library
)

#add_library(std-cxx-modules STATIC)
#set(MOD_BASE_DIR /lib/clang)
#target_sources(std-cxx-modules
#  PUBLIC
#    FILE_SET moduleStd
#    TYPE CXX_MODULES
#    BASE_DIRS ${MOD_BASE_DIR}
#    FILES
#      ${MOD_BASE_DIR}/std.cppm
#      ${MOD_BASE_DIR}/std.compat.cppm)
#target_compile_options(std-cxx-modules
#  PRIVATE
#    -Wno-reserved-module-identifier
#    -Wno-reserved-user-defined-literal)

#link_libraries(std-cxx-modules)

add_library(mmath)
target_sources(mmath
	PRIVATE	src/mmath.cpp
	PUBLIC
	  FILE_SET CXX_MODULES FILES src/mmath.cppm)

add_executable(main)
target_sources(main
  PRIVATE src/main.cpp)
target_link_libraries(main
  PRIVATE mmath)
