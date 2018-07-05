PREFIX=$DEVROOT/install/msi/
cmake .. \
      -DCMAKE_C_COMPILER=`which mpicc` \
      -DCMAKE_CXX_COMPILER=`which mpicxx` \
      -DCMAKE_Fortran_COMPILER=`which mpif90` \
      -DENABLE_COMPLEX=OFF \
      -DENABLE_TESTING=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -Dlas_DIR=$DEVROOT/install/las/lib/cmake \
      -DSCOREC_DIR=$DEVROOT/install/core/lib/cmake/SCOREC


#  -DSCOREC_LIB_DIR=$DEVROOT/install/core/lib \
    #  -DSCOREC_INCLUDE_DIR=$DEVROOT/install/core/include \
    #  -DZOLTAN_LIB_DIR=$DEVROOT/petsc/petsc-3.9.2/arch-linux2-c-debug/lib \
    #  -DPARMETIS_LIB_DIR=$DEVROOT/petsc/petsc-3.9.2/arch-linux2-c-debug/lib \

