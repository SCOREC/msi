cmake_minimum_required(VERSION 3.11)
project(msi VERSION 0.1.0 LANGUAGES CXX C Fortran)

include(cmake/bob.cmake)
bob_begin_package()

# make sure that the default is a RELEASE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Choose the type of build, options are: None Debug Release."
    FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# unless building shared libs, then select static libs
#  if both static and shared libs are available
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".so")
if(BUILD_SHARED_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(TARGET_LIB_NAME msi)

# find dependencies and set library and include dir info

find_package(SCOREC CONFIG REQUIRED)

if(ENABLE_PETSC)
  if(NOT PETSC_IMPLICIT_DEP)
    find_package(PETSc REQUIRED)
  endif()
  set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -DMSI_PETSC)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DMSI_PETSC)

  set(DEP_LIBS ${DEP_LIBS} ${PETSC_LIBRARIES})
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${PETSC_INCLUDE_DIRS})

  if(ENABLE_COMPLEX)
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -DPETSC_USE_COMPLEX)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DPETSC_USE_COMPLEX)
    set(TARGET_LIB_NAME ${TARGET_LIB_NAME}_complex)
  endif()
endif()

# source and header files

set(SOURCES
  api/msi.cc
  src/msi_solver.cc
  src/msi_petsc.cc
  )

set(HEADERS
  api/msi.h
  api/msi_types.h
  include/msi_solver.h
  include/msi_petsc.h
  )

# create the msi library and set include dirs and link libraries
add_library(${TARGET_LIB_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_LIB_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries(${TARGET_LIB_NAME} SCOREC::pumi SCOREC::apf ${DEP_LIBS})
target_include_directories(${TARGET_LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>
  $<INSTALL_INTERFACE:include/${TARGET_LIB_NAME}>
  ${INCLUDE_DIRS})

# installation and target export setup
bob_export_target(${TARGET_LIB_NAME})


# binary distribution package
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_VERSION "1.0.1")
include(CPack)

## Testing and Documentation

#Settings options for testing
enable_testing()
function(add_mpi_test tgt file np)
  message(STATUS "Adding test " ${tgt})
  add_executable(${tgt} ${file})
  target_link_libraries(${tgt} ${TARGET_LIB_NAME} ${DEP_LIBS})
  set(prms -np ${np} "${CMAKE_CURRENT_BINARY_DIR}/${tgt}" ${ARGN})
  add_test(NAME ${tgt} COMMAND "mpirun" ${prms})
endfunction(add_mpi_test)

include(CTest)
#This will be set to ON by the CTest driver script (and only by that)
option(ENABLE_TESTING "Build for CTest" OFF)
set(MPIRUN "mpirun"
  CACHE STRING
  "the mpirun or srun executable")
set(MPIRUN_PROCFLAG "-np"
  CACHE STRING
  "the command line flag to give process count to MPIRUN")

if(ENABLE_TESTING)
  add_mpi_test(msi_init test/init.cc 4)
  add_mpi_test(msi_init_subcomm test/init_subcomm 4)
  if(ENABLE_PETSC)
    add_executable(petsc test/petsc/main.cc)
    target_link_libraries(petsc ${TARGET_LIB_NAME} ${DEP_LIBS})
    # INSTALL(TARGETS petsc RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  endif()
endif()

# doxygen generation system
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

# finalize target export
bob_end_package()
