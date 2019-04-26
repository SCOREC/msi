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
/**
 *  @brief Set the communicator on which the linear system
 *         will operate.
 *  @note  Must be set after MPI_Init() but before
 *         PetscInitialize();
 */
void msi_matrix_setComm(MPI_Comm);
/**
 * @brief create a global parallel matrix or a local matrix
 *        structured to hold dofs from the provided field
 *        for a global parallel matrix, the rows owned
 *        correspond to the dofs on field nodes that are also
 *        owned, for a local matrix, all rows are owned, and
 *        there are rows for all dofs on the local process,
 *        regardless of ownership
 */
msi_matrix * msi_matrix_create(int matrix_type, pField f);
/**
 * @brief destroy the specified matrix
 */
void msi_matrix_delete(msi_matrix * mat);
/**
 * @brief destroy the specified matrix
 */
pField msi_matrix_getField(msi_matrix * mat);
/**
 * @brief finalize the assembly of the matrix, this must
 *        be done prior to a matrix solve or multiplication
 *        operation.
 */
void msi_matrix_assemble(msi_matrix * mat);
/**
 * @brief insert a single scalar value at the specified
 *        location in the matrix.
 */
void msi_matrix_insert(msi_matrix * m,
                       msi_int rw,
                       msi_int cl,
                       msi_scalar val);
/**
 * @brief add a single scalar value to the specified
 *        location in the matrix
 */
void msi_matrix_add(msi_matrix * mat,
                    msi_int rw,
                    msi_int cl,
                    msi_scalar val);
/**
 * @brief add a block of values ...
 */
void msi_matrix_addBlock(msi_matrix * mat,
                         pMeshEnt elem,
                         msi_int rowVarIdx,
                         msi_int columnVarIdx,
                         msi_scalar * vals);
/**
 * @brief set a dirichlet boundary condition on the
 *        specified row. This zeros all values in the
 *        row and sets a 1.0 on the diagonal.
 */
void msi_matrix_setBC(msi_matrix * mat, msi_int row);
/**
 * @brief set a laplacian boundary condition on the
 *        specified row, for the specified columns.
 *        this overwrites all existing values in the
 *        row;
 * @note the row and columns in this are specified
 *       using local indices rather than global indices
 */
void msi_matrix_setLaplaceBC(msi_matrix * mat,
                             msi_int row,
                             msi_int size,
                             msi_int * col,
                             msi_scalar * vals);

void msi_matrix_multiply(msi_matrix * mat,
                         pField inputvec,
                         pField outputvec);

void msi_matrix_solve(msi_matrix * mat,
                      pField rhs,
                      pField sol);
/**
 * @brief retrieve the number of iterations the last
 *        linear solve took to converge
 */
int msi_matrix_getNumIter(msi_matrix * mat);
void msi_matrix_write(msi_matrix * mat, const char* file_name, msi_int start_index = 0);
void msi_matrix_print(msi_matrix * mat);
#endif
