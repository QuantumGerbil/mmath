cmake_minimum_required(VERSION 3.28)

project(mmath LANGUAGES CXX)

# Ensure we use C++20 (or newer) with modules
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
# Ensure that we always compile with mold as linker and modules enabled.
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")
# Optional: disable compiler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# Force the final executable to appear in the project root:
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Create the executable target
add_executable(mmath)

# Collect regular .cpp files
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Collect .cppm files
file(GLOB MODULE_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cppm")

# Add both sets of sources to the executable
target_sources(mmath
  PRIVATE
    ${CPP_SOURCES}
    PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES
      BASE_DIRS "${CMAKE_SOURCE_DIR}/src"
      FILES 
	${MODULE_SOURCES}
)

# Enable the features needed for modules
target_compile_options(mmath 
		PRIVATE
		  -std=c++23)
