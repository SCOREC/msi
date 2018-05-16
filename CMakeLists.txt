#This is the top MSI CMakeList File for the Build

#Setting Version Number, Project Name
cmake_minimum_required (VERSION 3.1 FATAL_ERROR)
set(MSI_MAJOR 0)
set(MSI_MINOR 1)
set(MSI_PATCH 0)
set(MSI_VERSION ${MSI_MAJOR}.${MSI_MINOR}.${MSI_PATCH})
project(msi VERSION ${MSI_VERSION})
enable_language(C)
enable_language(CXX)
enable_language(Fortran)

# make sure that the default is a RELEASE
if(NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# default installation
#get_filename_component (default_prefix ".." ABSOLUTE)
#set (CMAKE_INSTALL_PREFIX ${default_prefix} CACHE STRING
#      "Choose the installation directory; by default it installs in the NORMA directory."
#      FORCE)

# FFLAGS depend on the compiler
get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
message(Fortran_COMPILER_NAME = ${Fortran_COMPILER_NAME})

#unless building shared libs, then select static libs
# if both static and shared libs are available
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".so")
if(BUILD_SHARED_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")
endif()

#Settings options for testing
enable_testing()
include(CTest)
#This will be set to ON by the CTest driver script (and only by that)
option(ENABLE_TESTING "Build for CTest" OFF)
set(MPIRUN "mpirun"
    CACHE string
    "the mpirun or srun executable")
set(MPIRUN_PROCFLAG "-np"
    CACHE string
    "the command line flag to give process count to MPIRUN")

#Doxygen generation system
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
               ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/msiConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/msiConfig.cmake")

set(CMAKE_MODULE_PATH
   ${CMAKE_MODULE_PATH}
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(lib_name msi)
find_package(Scorec REQUIRED)
set(include_dirs ${include_dirs} ${SCOREC_INCLUDE_DIRS})
set(link_libs ${link_libs} ${SCOREC_LIBRARIES})

find_package(las REQUIRED)
set(link_libs ${link_libs} las)

#Source and header files
set(SOURCES
  api/msi.cc
  src/msi_solver.cc
  src/msi_las.cc
)
set(HEADERS
  api/msi.h
  include/msi_solver.h
  include/msi_las.h
  )

add_library(${lib_name} ${SOURCES})
target_link_libraries(${lib_name} ${link_libs})
target_include_directories(${lib_name}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>
  $<INSTALL_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  ${include_dirs})

INSTALL(FILES ${HEADERS} DESTINATION include)
INSTALL(TARGETS ${lib_name}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

if (ENABLE_TESTING)
  add_executable(las_test test/las/main.cc)
  target_link_libraries(las_test ${lib_name} ${link_libs})
endif()

#binary distribution package
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_VERSION "${MSI_VERSION}")
include(CPack)
