cmake_minimum_required(VERSION 2.8)

SET(CTEST_DO_SUBMIT OFF)
SET(CTEST_TEST_TYPE Nightly)

set(USER "$ENV{USER}")

set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/fasttmp/${USER}/nightly")
set(CTEST_BUILD_CONFIGURATION "RelWithDebInfo")
set(CTEST_BUILD_FLAGS "-j 4")

set(CTEST_PROJECT_NAME "msi")
set(CTEST_SOURCE_NAME "msi")
set(CTEST_BINARY_NAME "build")

set(REPO_URL_BASE "git@github.com:SCOREC/msi")
set(BRANCHES "master;develop")
set(MERGE_AUTHOR "Nightly Bot <donotemail@scorec.rpi.edu>")

set(CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set(CTEST_BINARY_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/${CTEST_BINARY_NAME}")
set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION} "tmpnam")

find_program(CTEST_GIT_COMMAND NAMES git)
set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

function(setup_repo)
  if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/")
    message("Running \"git clone ${REPO_URL_BASE}.git ${CTEST_SOURCE_DIRECTORY}\"")
    execute_process(COMMAND "${CTEST_GIT_COMMAND}" clone "${REPO_URL_BASE}.git" "${CTEST_SOURCE_DIRECTORY}" RESULT_VARIABLE CLONE_RET)
    if(CLONE_RET)
      message(FATAL_ERROR "Cloning ${REPO_URL_BASE}.git failed (code ${RETVAR})!")
    else()
      message("Cloning ${REPO_URL_BASE}.git succeeded")
    endif()
    # make local tracking versions of all remote branches
    foreach(BRANCH IN LISTS BRANCHES)
      if(NOT "${BRANCH}" STREQUAL "master")
        create_branch(${BRANCH} origin/${BRANCH})
      endif()
    endforeach()
  endif()
endfunction(setup_repo)

function(setup_meshes)
  execute_process(COMMAND rm -rf "${MESHES}"
        WORKING_DIRECTORY "${SCOREC}"
        RESULT_VARIABLE RM_RET)
  execute_process(COMMAND "${CTEST_GIT_COMMAND}" clone ${MESH_URL_BASE}.git meshes
        WORKING_DIRECTORY "${SCOREC}"
        RESULT_VARIABLE RETVAR)
  if(RETVAR)
    message(FATAL_ERROR "failed to clone meshes repository")
  else()
    message("check out meshes repository")
  endif()
endfunction(setup_meshes)

function(git_exec CMD ACTION)
  string(REPLACE " " ";" CMD2 "${CMD}")
  message("Running \"git ${CMD}\"")
  execute_process(COMMAND "${CTEST_GIT_COMMAND}" ${CMD2}
    WORKING_DIRECTORY "${CTEST_SOURCE_DIRECTORY}"
    RESULT_VARIABLE RETVAR)
  if(RETVAR)
    message(FATAL_ERROR "${ACTION} failed (code ${RETVAR})!")
  else()
    message("${ACTION} succeeded")
  endif()
endfunction(git_exec)

function(create_branch BRANCH_NAME TRACKING_NAME)
  git_exec("branch --track ${BRANCH_NAME} ${TRACKING_NAME}"
           "Creating branch ${BRANCH_NAME}")
endfunction(create_branch)

function(checkout_branch BRANCH_NAME)
  git_exec("checkout ${BRANCH_NAME}"
           "Checking out branch ${BRANCH_NAME}")
endfunction(checkout_branch)

function(setup_repo)
  if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
    message("Running \"git clone ${REPO_URL_BASE}.git ${CTEST_SOURCE_DIRECTORY}\"")
    execute_process(COMMAND "${CTEST_GIT_COMMAND}" clone ${REPO_URL_BASE}.git "${CTEST_SOURCE_DIRECTORY}" RESULT_VARIABLE CLONE_RET)
    if(CLONE_RET)
      message(FATAL_ERROR "Cloning ${REPO_URL_BASE}.git failed (code ${RETVAR})!")
    else()
      message("Cloning ${REPO_URL_BASE}.git succeeded")
    endif()
    # make local tracking versions of all remote branches
    foreach(BRANCH IN LISTS BRANCHES)
      if(NOT "${BRANCH}" STREQUAL "master")
        create_branch(${BRANCH} origin/${BRANCH})
      endif()
    endforeach()
  endif()
endfunction(setup_repo)

function(check_current_branch BRANCH_NAME CONFIG_OPTS ERRVAR)
  file(MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")

  ctest_configure(
      BUILD "${CTEST_BINARY_DIRECTORY}"
      SOURCE "${CTEST_SOURCE_DIRECTORY}"
      OPTIONS "${CONFIG_OPTS}"
      RETURN_VALUE CONFIG_RET)
  if(CONFIG_RET)
    message(WARNING "${BRANCH_NAME} config failed (code ${CONFIG_RET})!")
  else()
    message("${BRANCH_NAME} config passed")
  endif()

  ctest_build(
      BUILD "${CTEST_BINARY_DIRECTORY}"
      NUMBER_ERRORS NUM_BUILD_ERRORS
      NUMBER_WARNINGS NUM_BUILD_WARNINGS
      RETURN_VALUE BUILD_RET)
  if(NUM_BUILD_WARNINGS OR
      NUM_BUILD_ERRORS OR BUILD_RET)
    message(WARNING "
${BRANCH_NAME} build failed!
  ${NUM_BUILD_WARNINGS} warnings
  ${NUM_BUILD_ERRORS} errors
  code ${BUILD_RET}")
  else()
    message("${BRANCH_NAME} build passed")
  endif()

  ctest_test(
      BUILD "${CTEST_BINARY_DIRECTORY}"
      RETURN_VALUE TEST_RET)
  if(TEST_RET)
    message(WARNING "${BRANCH_NAME} testing failed (code ${TEST_RET})!")
  else()
    message("${BRANCH_NAME} testing passed")
  endif()

  if(CONFIG_RET OR
     NUM_BUILD_WARNINGS OR
     NUM_BUILD_ERRORS OR BUILD_RET OR
     TEST_RET)
    message(WARNING "some ${BRANCH_NAME} checks failed!")
    set(${ERRVAR} True PARENT_SCOPE)
  else()
    message("all ${BRANCH_NAME} checks passed")
    set(${ERRVAR} False PARENT_SCOPE)
  endif()

  if(CTEST_DO_SUBMIT)
    ctest_submit(PARTS Update Configure Build Test
        RETRY_COUNT 4
        RETRY_DELAY 30
        RETURN_VALUE SUBMIT_ERROR)
    if(SUBMIT_ERROR)
      message(WARNING "Could not submit ${BRANCH_NAME} results to CDash (code ${SUBMIT_ERROR})!")
    else()
      message("Submitted ${BRANCH_NAME} results to CDash")
    endif()
  endif()
endfunction(check_current_branch)

function(check_tracking_branch BRANCH_NAME CONFIG_OPTS ERRVAR)
  checkout_branch("${BRANCH_NAME}")
  set_property(GLOBAL PROPERTY SubProject "${BRANCH_NAME}")
  set_property(GLOBAL PROPERTY Label "${BRANCH_NAME}")
  ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}" RETURN_VALUE NUM_UPDATES)
  if("${NUM_UPDATES}" EQUAL "-1")
    message(FATAL_ERROR "Could not update ${BRANCH_NAME} branch!")
  endif()
  message("Updated ${NUM_UPDATES} files")
  check_current_branch(${BRANCH_NAME} "${CONFIG_OPTS}" ERRVAL2)
  set(${ERRVAR} ${ERRVAL2} PARENT_SCOPE)
endfunction(check_tracking_branch)

function(check_merge_branch BRANCH_NAME CONFIG_OPTS ERRVAR)
  set_property(GLOBAL PROPERTY SubProject "${BRANCH_NAME}")
  set_property(GLOBAL PROPERTY Label "${BRANCH_NAME}")
  check_current_branch(${BRANCH_NAME} "${CONFIG_OPTS}" ERRVAL2)
  set(${ERRVAR} ${ERRVAL2} PARENT_SCOPE)
endfunction(check_merge_branch)

function(update_branch BRANCH_NAME)
  checkout_branch(${BRANCH_NAME})
  git_exec("pull --ff-only"
           "Fast-forward pulling ${BRANCH_NAME}")
endfunction(update_branch)

function(start_merge FIRST_BRANCH SECOND_BRANCH NEXT_ACTION)
  update_branch(${FIRST_BRANCH})
  update_branch(${SECOND_BRANCH})
  set(NEW_BRANCH "${SECOND_BRANCH}-into-${FIRST_BRANCH}")
  create_branch(${NEW_BRANCH} origin/${FIRST_BRANCH})
  checkout_branch(${NEW_BRANCH})
  message("Running \"git merge --no-ff --no-commit ${SECOND_BRANCH}\"")
  execute_process(COMMAND "${CTEST_GIT_COMMAND}" merge --no-ff --no-commit ${SECOND_BRANCH}
    WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}
    OUTPUT_VARIABLE MERGE_OUTPUT
    RESULT_VARIABLE MERGE_RET)
  if("${MERGE_OUTPUT}" MATCHES "CONFLICT")
    message(WARNING "Merging ${SECOND_BRANCH} into ${FIRST_BRANCH} causes conflicts!")
    set(${NEXT_ACTION} ABORT PARENT_SCOPE)
    return()
  endif()
  if("${MERGE_OUTPUT}" MATCHES "Already up-to-date")
    message("${FIRST_BRANCH} up-to-date with ${SECOND_BRANCH}, stopping merge")
    set(${NEXT_ACTION} CLEANUP PARENT_SCOPE)
    return()
  endif()
  if(MERGE_RET)
    message(FATAL_ERROR "Merging ${SECOND_BRANCH} into ${FIRST_BRANCH} failed (code ${MERGE_RET})!")
  endif()
  message("Merging ${SECOND_BRANCH} into ${FIRST_BRANCH} worked okay...")
  set(${NEXT_ACTION} PROCEED PARENT_SCOPE)
endfunction(start_merge)

function(cleanup_merge FIRST_BRANCH SECOND_BRANCH)
  set(NEW_BRANCH "${SECOND_BRANCH}-into-${FIRST_BRANCH}")
  checkout_branch(master)
  git_exec("branch -D ${NEW_BRANCH}"
           "Deleting temporary branch ${NEW_BRANCH}")
endfunction(cleanup_merge)

function(accept_merge FIRST_BRANCH SECOND_BRANCH)
  set(NEW_BRANCH "${SECOND_BRANCH}-into-${FIRST_BRANCH}")
  message("Running \"git commit -m \"Merging ${SECOND_BRANCH} into ${FIRST_BRANCH}\" --author=\"${MERGE_AUTHOR}\"\"")
  execute_process(COMMAND "${CTEST_GIT_COMMAND}" commit
    -m "Merging ${SECOND_BRANCH} into ${FIRST_BRANCH}"
    --author="${MERGE_AUTHOR}"
    WORKING_DIRECTORY "${CTEST_SOURCE_DIRECTORY}"
    RESULT_VARIABLE RETVAR)
  if(RETVAR)
    message(FATAL_ERROR "Commiting merge ${NEW_BRANCH} failed (code ${RETVAR})!")
  else()
    message("Commiting merge ${NEW_BRANCH} succeeded")
  endif()
  git_exec("push origin ${NEW_BRANCH}:${FIRST_BRANCH}"
           "Pushing merge ${NEW_BRANCH}")
  cleanup_merge(${FIRST_BRANCH} ${SECOND_BRANCH})
endfunction(accept_merge)

function(abort_merge FIRST_BRANCH SECOND_BRANCH)
  set(NEW_BRANCH "${SECOND_BRANCH}-into-${FIRST_BRANCH}")
  git_exec("merge --abort"
           "Aborting ${NEW_BRANCH} merge")
  cleanup_merge(${FIRST_BRANCH} ${SECOND_BRANCH})
endfunction(abort_merge)

function(try_merge FIRST_BRANCH SECOND_BRANCH CONFIG)
  start_merge(${FIRST_BRANCH} ${SECOND_BRANCH} NEXT_ACTION)
  if("${NEXT_ACTION}" STREQUAL "CLEANUP")
    cleanup_merge(${FIRST_BRANCH} ${SECOND_BRANCH})
    return()
  elseif("${NEXT_ACTION}" STREQUAL "ABORT")
    abort_merge(${FIRST_BRANCH} ${SECOND_BRANCH})
    return()
  endif()
  set(NEW_BRANCH "${SECOND_BRANCH}-into-${FIRST_BRANCH}")
  check_merge_branch("${NEW_BRANCH}" "${CONFIG}" CHECK_ERR)
  if(CHECK_ERR)
    abort_merge(${FIRST_BRANCH} ${SECOND_BRANCH})
    return()
  endif()
  accept_merge(${FIRST_BRANCH} ${SECOND_BRANCH})
endfunction(try_merge)

# Main code !
ctest_start(${CTEST_TEST_TYPE})

SET(CONFIGURE_OPTIONS
  "-DCMAKE_C_COMPILER:FILEPATH=mpicc"
  "-DCMAKE_CXX_COMPILER:FILEPATH=mpicxx"
  "-DCMAKE_Fortran_COMPILER:FILEPATH=mpif90"
  "-DSCOREC_DIR:FILEPATH=/fasttmp/${USER}/dev/install/core/lib/cmake/SCOREC/"
  "-DENABLE_PETSC:BOOL=ON"
  "-DPETSC_DIR:FILEPATH=$ENV{PETSC_DIR}"
  "-DPETSC_ARCH:FILEPATH=$ENV{PETSC_ARCH}"
  "-DENABLE_COMPLEX:BOOL=OFF"
  "-DENABLE_TESTING:BOOL=ON"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMAKE_INSTALL_PREFIX:FILEPATH=/fasttmp/${USER}/nightly/install/msi"
  )

setup_repo()

foreach(BRANCH IN LISTS BRANCHES)
  check_tracking_branch("${BRANCH}" "${CONFIGURE_OPTIONS}" CHECK_ERR)
endforeach()

try_merge(master develop "${CONFIGURE_OPTIONS}" ${ALLOWED_WARNINGS})
