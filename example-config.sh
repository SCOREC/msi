if [ -z "${DEVROOT}" ] ; then
    echo "Please set DEVROOT in your environment"
    exit -1
fi

if [ -z "${INSTALLROOT}" ] ; then
    INSTALLROOT=$DEVROOT/install
fi

if [ -z "${SCOREC_ROOT}" ] ; then
    SCOREC_ROOT=$INSTALLROOT/core/lib/cmake/SCOREC
fi

if [ -z "${PETSC_DIR}" ] ; then
    echo "No PETSC_DIR in the environment!"
    echo "module load petsc..."
    module load petsc
    if [ $? == 0 ] ; then
        echo " succeeded!"
    else
        echo " failed!"
        exit -1
    fi
fi

PREFIX=$INSTALLROOT/msi

cmake .. \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_Fortran_COMPILER=mpif90 \
      -DSCOREC_DIR=$SCOREC_ROOT \
      -DENABLE_PETSC=ON \
      -DPETSC_DIR=$PETSC_DIR \
      -DPETSC_ARCH=$PETSC_ARCH \
      -DENABLE_COMPLEX=OFF \
      -DENABLE_TESTING=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=$PREFIX

