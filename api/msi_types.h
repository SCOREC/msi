#ifndef MSI_TYPES_H_
#define MSI_TYPES_H_

// backend-dependent types
#include <petscsystypes.h>

using msi_int = PetscInt;
using msi_scalar = PetscScalar;

class msi_matrix;
using pMatrix = msi_matrix*;

// native MSI types and symbols
#define MSI_SUCCESS 0
#define MSI_FAILURE 1

enum msi_matrix_type
{
  MSI_MULTIPLY = 0,
  MSI_SOLVE = 1
};

enum msi_matrix_status
{
  MSI_NOT_FIXED = 0,
  MSI_FIXED = 1
};

#endif
