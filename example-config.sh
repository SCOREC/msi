# edit these params
DEVLOC=fasttmp
DEVDIR=dev
CC=mpicc
CXX=mpicx
FTN=mpif90
IMPLICIT_PETSC=OFF

# derived vars
USER=`whoami`
DEVROOT=${DEVLOC}/${USER}/${DEVDIR}

if [ -z "${INSTALLROOT}" ] ; then
    echo "No INSTALLROOT in environment, default is $DEVROOT/install"
    INSTALLROOT=$DEVROOT/install
fi

if [ -z "${SCOREC_ROOT}" ] ; then
    echo "No SCOREC_ROOT in environment, default is $INSTALLROOT/core/lib/cmake/SCOREC"
    SCOREC_ROOT=$INSTALLROOT/core/lib/cmake/SCOREC
fi

if [ -z "${PETSC_DIR}" ] ; then
    echo "No PETSC_DIR in the environment!"
    echo "trying module load petsc..."
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
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_Fortran_COMPILER=$FTN \
      -DSCOREC_DIR=$SCOREC_ROOT \
      -DENABLE_PETSC=ON \
      -DPETSC_IMPLICIT_DEP=$IMPLICIT_PETSC \
      -DENABLE_COMPLEX=OFF \
      -DENABLE_TESTING=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$PREFIX

