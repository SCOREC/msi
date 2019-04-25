#ifndef MSI_TYPES_H_
#define MSI_TYPES_H_

#include <petscsystypes.h>

using msi_int = PetscInt;
using msi_scalar = PetscScalar;

class msi_matrix;
using pMatrix = msi_matrix*;

#endif
