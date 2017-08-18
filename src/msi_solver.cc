/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifdef MSI_PETSC
#include "msi.h"
#include "msi_petsc.h"
#include "apf.h"
#include "apfNumbering.h"
#include "apfShape.h"
#include "apfMesh.h"
#include <vector>
#include <set>
#include "PCU.h"
#include <assert.h>
#include <iostream>

#ifdef PETSC_USE_COMPLEX
#include "petscsys.h" // for PetscComplex
#include <complex>
using std::complex;
#endif

// ***********************************
// 		MSI_SOLVER
// ***********************************

msi_solver* msi_solver::_instance=NULL;
msi_solver* msi_solver::instance()
{
  if (_instance==NULL)
    _instance = new msi_solver();
  return _instance;
}

msi_solver::msi_solver()
: assembleOption(0) 
{
  matrix_container = new std::map<int, msi_matrix*>;
}

msi_solver::~msi_solver()
{
  if (matrix_container!=NULL)
    matrix_container->clear();
  matrix_container=NULL;
  delete _instance;
}

void msi_solver::add_matrix(int matrix_id, msi_matrix* matrix)
{
  assert(matrix_container->find(matrix_id)==matrix_container->end());
  matrix_container->insert(std::map<int, msi_matrix*>::value_type(matrix_id, matrix));
}

msi_matrix* msi_solver::get_matrix(int matrix_id)
{
  std::map<int, msi_matrix*>::iterator mit = matrix_container->find(matrix_id);
  if (mit == matrix_container->end()) 
    return (msi_matrix*)NULL;
  return mit->second;
}
#endif
