#ifndef MSI_PETSC_VERSION_H_
#define MSI_PETSC_VERSION_H_
#include <petscversion.h>
#include <petsc.h>
#include <utility>

template <typename ... Args>
auto PetscSetPCBackend(Args&&... args) -> decltype(PCFactorSetMatSolverType(std::forward<Args>(args)...))
{
#if PETSC_VERSION_GT(3,9,5)
  return PCFactorSetMatSolverType(std::forward<Args>(args)...);
#else
  return PCFactorSetMatSolverPackage(std::forward<Args>(args)...);
#endif
}
#endif
