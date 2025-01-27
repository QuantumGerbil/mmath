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
add_compile_options(-fdiagnostics-color -pedantic-errors -Wall -Wextra -Wdeprecated -fexceptions)
add_link_options(-fuse-ld=lld -Wl,--color-diagnostics -fdiagnostics-color)

file(GLOB MODULE_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cppm")
file(GLOB SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_library(mmath)
target_sources(mmath
	PUBLIC
	  FILE_SET CXX_MODULES
	  FILES 
#		src/mmath.cppm 
#		src/mmath_vector.cppm
		${MODULE_FILES}
)

add_executable(main)
target_sources(main
  PRIVATE src/main.cpp)
target_link_libraries(main
  PRIVATE mmath)
