/******************************************************************************

  (c) 2017 - 2019 Scientific Computation Research Center,
      Rensselaer Polytechnic Institute. All rights reserved.

  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.

*******************************************************************************/
#ifndef MSI_API_H_
#define MSI_API_H_
#include "msi_types.h"
#include "msi_sync.h" // todo : require that this is explicitly included by the user since this contains developer-only functions
#include <pumi.h>
#include <mpi.h>
void msi_init(int argc, char * argv[], MPI_Comm cm);
// remember to delete ownership after use
void msi_start(pMesh m,
               pOwnership o = NULL,
               pShape s = NULL,
               MPI_Comm cm = MPI_COMM_NULL);
void msi_stop(pMesh m);
void msi_finalize();
pOwnership msi_getOwnership( );
pNumbering msi_numbering_createGlobal_multiOwner(pMesh m,
                                                 const char* name,
                                                 pShape s,
                                                 pOwnership o,
                                                 MPI_Comm cm);  // [PARASOL]
// field creation with multiple variables
pField msi_field_create(pMesh m,
                        const char* /* in */ field_name,
                        int /*in*/ nv,
                        int /*in*/ nd,
                        pShape shape = NULL);
int msi_field_getNumVal(pField f);
int msi_field_getSize(pField f);
// returns sequential local numbering of entity's ith node
// local numbering is based on mesh shape
int msi_node_getID(pMeshEnt e, int n);
// returns global numbering of entity's ith node
// global numbering is based on ownership set in msi_start
int msi_node_getGlobalID(pMeshEnt e, int n);
void msi_node_setField(pField f,
                       pMeshEnt e,
                       int n,
                       int size_dof,
                       double* dof_data);
int msi_node_getField(pField f, pMeshEnt e, int n, double* dof_data);
// returns local DOF id range based on local numbering
void msi_node_getFieldID(pField f,
                         pMeshEnt e,
                         int n,
                         int* /* out */ start_dof_id,
                         int* /* out */ end_dof_id_plus_one);
// returns global DOF id range based on ownership
void msi_node_getGlobalFieldID(pField f,
                               pMeshEnt e,
                               int n,
                               int* /* out */ start_dof_id,
                               int* /* out */ end_dof_id_plus_one);
/*
 * Set the communicator on which the linear system will operate.
 * Must be set after MPI_Init() but before PetscInitialize();
 */
void msi_matrix_setComm(MPI_Comm);
/** matrix and solver functions with PETSc */
msi_matrix * msi_matrix_create(int matrix_type, pField f);
void msi_matrix_delete(msi_matrix * mat);
pField msi_matrix_getField(msi_matrix * mat);
void msi_matrix_assemble(msi_matrix * mat);
void msi_matrix_insert(msi_matrix * mat,
                       msi_int row,
                       msi_int column,
                       int scalar_type,
                       double* val);
void msi_matrix_add(msi_matrix * mat,
                    msi_int row,
                    msi_int column,
                    int scalar_type,
                    double* val);
void msi_matrix_addBlock(msi_matrix * mat,
                         pMeshEnt elem,
                         msi_int rowVarIdx,
                         msi_int columnVarIdx,
                         double* values);
void msi_matrix_setBC(msi_matrix * mat, msi_int row);
void msi_matrix_setLaplaceBC(msi_matrix * mat,
                             msi_int row,
                             msi_int size,
                             msi_int* columns,
                             double* values);
void msi_matrix_multiply(msi_matrix * mat, pField inputvec, pField outputvec, bool sync = true);
void msi_AxpBy(msi_matrix * A, pField x, msi_matrix * B, pField y, pField z);
void msi_matrix_solve(msi_matrix * mat, pField rhs, pField sol, bool sync = true);
int msi_matrix_getNumIter(msi_matrix * mat);
// auxiliary
void msi_matrix_write(msi_matrix * mat, const char* file_name, msi_int start_index = 0);
void msi_matrix_print(msi_matrix * mat);
#endif
