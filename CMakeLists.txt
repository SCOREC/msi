cmake_minimum_required(VERSION 3.11)
project(msi)
enable_language(CXX)
enable_language(C)
enable_language(Fortran)

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

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/msiConfig.cmake.in"
  "${CMAKE_BINARY_DIR}/msiConfig.cmake")

# source and header files
set(SOURCES
  api/msi.cc
  src/msi_solver.cc
  src/msi_petsc.cc
  src/msi_trilinos.cc
)

set(HEADERS
  api/msi.h
  include/msi_solver.h
  include/msi_petsc.h
  include/msi_trilinos.h
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(TARGET_LIB_NAME msi)

# find dependencies and set library and include dir info

find_package(SCOREC CONFIG REQUIRED)

if(ENABLE_TRILINOS)
  find_package(Trilinos REQUIRED)
  find_package(Boost REQUIRED)

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMSI_TRILINOS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMSI_TRILINOS")

  set(INCLUDE_DIRS ${TRILINOS_INCLUDE_DIRS})
  set(DEP_LIBS ${DEP_LIBS} ${TRILINOS_LIBRARIES} ${BOOST_LIBRARIES})

  set(TARGET_LIB_NAME ${TARGET_LIB_NAME}_trilinos)
else()
  set(ENABLE_PETSC "ON")
endif()

if(ENABLE_PETSC)
  find_package(PETSc REQUIRED)
  set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -DMSI_PETSC)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DMSI_PETSC)

  set(DEP_LIBS ${DEP_LIBS} ${PETSC_LIBRARIES})
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${PETSC_INCLUDE_DIRS})

  find_package(HDF5 REQUIRED)
  set(DEP_LIBS ${DEP_LIBS} ${HDF5_LIBRARIES})
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})

  set(TARGET_LIB_NAME ${TARGET_LIB_NAME}_petsc)

  if(ENABLE_COMPLEX)
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -DPETSC_USE_COMPLEX)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DPETSC_USE_COMPLEX)
    set(TARGET_LIB_NAME ${TARGET_LIB_NAME}_complex)
  endif()
endif()

# create the msi library and set include dirs and link libraries

add_library(${TARGET_LIB_NAME} STATIC ${SOURCES})
target_link_libraries(${TARGET_LIB_NAME} SCOREC::pumi SCOREC::apf ${DEP_LIBS})
target_include_directories(${TARGET_LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>
  $<INSTALL_INTERFACE:include/${TARGET_LIB_NAME}>
  ${INCLUDE_DIRS})

# installation instructions

INSTALL(FILES ${HEADERS} DESTINATION include)
INSTALL(TARGETS ${TARGET_LIB_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include/${TARGET_LIB_NAME})

# binary distribution package
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_VERSION "1.0.1")
include(CPack)

## Testing and Documentation

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

if(ENABLE_TESTING)
  if(ENABLE_PETSC)
    add_executable(petsc test/petsc/main.cc)
    target_link_libraries(petsc ${TARGET_LIB_NAME} ${DEP_LIBS})
    # INSTALL(TARGETS petsc RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  endif()
  if(ENABLE_TRILINOS)
    add_executable(epetra test/epetra/main.cc)
    target_link_libraries(epetra ${TARGET_LIB_NAME} ${DEP_LIBS})
    # INSTALL(TARGETS epetra RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  endif()
endif()

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

