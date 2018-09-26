PROJECT="msi"
REPO="git@github.com:SCOREC/msi.git"

USER=`whoami`
WWW="/net/web/public/${USER}/nightly/cdash/${PROJECT}/cmake.log"
DEVROOT="/fasttmp/${USER}/nightly/"

#PWD=`pwd`
#[[ ! "${PWD}" =~ "${DEVROOT}" ]] && "nightly.sh script is being run in the incorrect directory, exiting..." && exit -1

export PATH=/usr/share/lmod/lmod/libexe:${PATH}
unset MODULEPATH
module use /opt/scorec/spack/lmod/linux-rhel7-x86_64/Core/
module use /opt/scorec/modules/
module load git
module load cmake
module load gcc
module load mpich
module load petsc/3.9.3-int32-hdf5+ftn-real-c-meo4jde

project_root=${DEVROOT}/${PROJECT}
build_dir=${project_root}/build
nightly_dir=${project_root}/cdash

[[ ! -d ${project_root} ]] && git clone "${REPO}" "${DEVROOT}/${PROJECT}"
cd ${project_root} && git checkout dev && git pull

git rev-parse --verify dev-into-master
[[ $? == 0 ]] && git branch -d dev-into-master

[[ -d ${build_dir} ]] && rm -rf ${build_dir}
mkdir ${build_dir}

cp ${nightly_dir}/CTestConfig.cmake ${build_dir}/CTestConfig.cmake
cp ${project_root}/example-config.sh ${build_dir}/config.sh
cp ${nightly_dir}/scorec-nightly.patch ${build_dir}/scorec-nightly.patch
patch ${build_dir}/config.sh ${build_dir}/scorec-nightly.patch

cd ${build_dir}
rm CMakeCache.txt
./config.sh

ctest -VV --output-on-failue --script ${nightly_dir}/nightly.cmake &> cmake.log
cp cmake.log ${WWW}

cd ${build_dir}
git checkout dev
rm CMakeCache.txt
./config.sh
make install


