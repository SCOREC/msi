#This is the top MSI CMakeList File for the Build

#Setting Version Number, Project Name
cmake_minimum_required (VERSION 2.8)
project (msi)
enable_language (Fortran)

# make sure that the default is a RELEASE
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release."
      FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# default installation
#get_filename_component (default_prefix ".." ABSOLUTE)
#set (CMAKE_INSTALL_PREFIX ${default_prefix} CACHE STRING
#      "Choose the installation directory; by default it installs in the NORMA directory."
#      FORCE)

# FFLAGS depend on the compiler
get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
message (Fortran_COMPILER_NAME = ${Fortran_COMPILER_NAME})

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

#Source and header files
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

set(CMAKE_MODULE_PATH 
   ${CMAKE_MODULE_PATH} 
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api)
set(TARGET_LIB_NAME msi)

find_package(Scorec QUIET REQUIRED)
include_directories(${SCOREC_INCLUDE_DIRS})
set(DEP_LIBS ${DEP_LIBS} ${SCOREC_LIBRARIES} )

if (ENABLE_TRILINOS)
find_package(Trilinos QUIET REQUIRED)
find_package(Boost QUIET REQUIRED)
include_directories(${TRILINOS_INCLUDE_DIRS})
set(DEP_LIBS ${DEP_LIBS} ${TRILINOS_LIBRARIES})
set(DEP_LIBS ${DEP_LIBS} ${BOOST_LIBRARIES})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMSI_TRILINOS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMSI_TRILINOS")
set(TARGET_LIB_NAME ${TARGET_LIB_NAME}_trilinos)
else()
  set(ENABLE_PETSC "ON")
endif()

if (ENABLE_PETSC)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMSI_PETSC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMSI_PETSC")
find_package(Hdf5 QUIET REQUIRED)
set(DEP_LIBS ${DEP_LIBS} ${HDF5_LIBRARIES} )
find_package(Petsc QUIET REQUIRED)
include_directories(${PETSC_INCLUDE_DIRS})
set(DEP_LIBS ${DEP_LIBS} ${PETSC_LIBRARIES} )
endif()

if(ENABLE_COMPLEX)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPETSC_USE_COMPLEX")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPETSC_USE_COMPLEX")
set(TARGET_LIB_NAME ${TARGET_LIB_NAME}_complex)
endif()

add_library(${TARGET_LIB_NAME} ${SOURCES})
target_link_libraries(${TARGET_LIB_NAME} ${DEP_LIBS})

INSTALL(FILES ${HEADERS} DESTINATION include)
INSTALL(TARGETS ${TARGET_LIB_NAME}
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib)

if (ENABLE_TESTING)
if (ENABLE_PETSC)
add_executable(petsc test/petsc/main.cc)
target_link_libraries(petsc ${DEP_LIBS})
target_link_libraries(petsc ${TARGET_LIB_NAME})
INSTALL(TARGETS petsc RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)  
endif()
if (ENABLE_TRILINOS)
    add_executable(epetra test/epetra/main.cc)
    target_link_libraries(epetra ${DEP_LIBS})
    target_link_libraries(epetra ${TARGET_LIB_NAME})
    INSTALL(TARGETS epetra RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()
endif() #enable_testing

#binary distribution package
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_VERSION "1.0.1")
include(CPack)

message("ENV_FLAGS = $ENV{FFLAGS}")
message("CMAKE_C_FLAGS = ${CMAKE_C_FLAGS}")
message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message("CMAKE_Fortran_COMPILER_INIT = ${CMAKE_Fortran_COMPILER_INIT}")
message("CMAKE_Fortran_COMPILER_FULLPATH =
${CMAKE_Fortran_COMPILER_FULLPATH}")
message("CMAKE_Fortran_COMPILER = ${CMAKE_Fortran_COMPILER}")
message("CMAKE_Fortran_FLAGS = ${CMAKE_Fortran_FLAGS}")
message("CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")


