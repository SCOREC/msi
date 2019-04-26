/******************************************************************************

  (c) 2017 - 2019 Scientific Computation Research Center,
      Rensselaer Polytechnic Institute. All rights reserved.

  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.

*******************************************************************************/
#ifndef MSI_PETSC_VERSION_H_
#define MSI_PETSC_VERSION_H_
#include <petscversion.h>
#include <petsc.h>
#include <utility>

template <typename ... Args>
auto PetscSetPCBackend(Args&&... args) -> decltype(
#if PETSC_VERSION_GT(3,9,5)
  PCFactorSetMatSolverType(std::forward<Args>(args)...)
#else
  PCFactorSetMatSolverPackage(std::forward<Args>(args)...)
#endif
  )
{
#if PETSC_VERSION_GT(3,9,5)
  return PCFactorSetMatSolverType(std::forward<Args>(args)...);
#else
  return PCFactorSetMatSolverPackage(std::forward<Args>(args)...);
#endif
}
#endif
