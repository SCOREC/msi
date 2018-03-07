/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifndef MSI_HEADER_H
#define MSI_HEADER_H
#include "pumi.h"

// to-delete
#define MSI_SUCCESS 0
#define MSI_FAILURE 1

// end of to-delete
enum msi_matrix_type { /*0*/ MSI_MULTIPLY=0, 
                       /*1*/ MSI_SOLVE}; 
enum msi_matrix_status { /*0*/ MSI_NOT_FIXED=0,
                         /*1*/ MSI_FIXED};

// START OF API
// remember to delete ownership after use
void msi_start(pMesh m, pOwnership o=NULL, pShape s=NULL);
void msi_finalize(pMesh m);
pOwnership msi_getOwnership();

// field creation with multiple variables
pField msi_field_create (pMesh m, const char* /* in */ field_name, 
                      int /*in*/ nv, int /*in*/ nd, pShape shape=NULL);
int msi_field_getNumVal(pField f);
int msi_field_getSize(pField f);

// returns sequential local numbering of entity's ith node
// local numbering is based on mesh shape 
int msi_node_getID (pMeshEnt e, int n);

// returns global numbering of entity's ith node 
// global numbering is based on ownership set in msi_start
int msi_node_getGlobalID (pMeshEnt e, int n);

void msi_node_setField (pField f, pMeshEnt e, int n, int size_dof, double* dof_data);
int msi_node_getField(pField f, pMeshEnt e, int n, double* dof_data);

// returns local DOF id range based on local numbering
void msi_node_getFieldID (pField f, pMeshEnt e, int n,
     int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);

// returns global DOF id range based on ownership
void msi_node_getGlobalFieldID (pField f, pMeshEnt e, int n,
     int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);

#ifdef MSI_PETSC
#include "msi_petsc.h"
class msi_matrix;
typedef msi_matrix* pMatrix;

/*
 * Set the communicator on which the linear system will operate.
 * Must be set after MPI_Init() but before PetscInitialize();
 */
void msi_set_comm_comm(MPI_Comm);

/** matrix and solver functions with PETSc */
pMatrix msi_matrix_create(int matrix_type, pField f);
void msi_matrix_delete(pMatrix mat);
pField msi_matrix_getField(pMatrix mat);

void msi_matrix_assemble(pMatrix mat);

void msi_matrix_insert(pMatrix mat, int row, int column, int scalar_type, double* val);
void msi_matrix_add(pMatrix mat, int row, int column, int scalar_type, double* val);
void msi_matrix_addBlock(pMatrix mat, pMeshEnt elem, int rowVarIdx, int columnVarIdx, double* values);

void msi_matrix_setBC(pMatrix mat, int row);
void msi_matrix_setLaplaceBC (pMatrix mat, int row, int size, int* columns, double* values);

void msi_matrix_multiply(pMatrix mat, pField inputvec, pField outputvec);

void msi_matrix_solve(pMatrix mat, pField rhs, pField sol);
int msi_matrix_getNumIter(pMatrix mat);

// auxiliary
void msi_matrix_write(pMatrix mat, const char* file_name, int start_index=0);
void msi_matrix_print(pMatrix mat);
#endif // #ifdef MSI_PETSC

#ifdef MSI_TRILINOS
//=========================================================================
/** matrix and solver functions with TRILINOS */
//=========================================================================
int msi_epetra_create(int* matrix_id, int* matrix_type, pField field);
int msi_epetra_delete(int* matrix_id);

int msi_epetra_insert(int* matrix_id, int* row, int* column, double* val);
int msi_epetra_addblock(int* matrix_id, int * ielm, int* rowVarIdx, int * columnVarIdx, double * values);

int msi_epetra_setbc(int* matrix_id, int* row);
int msi_epetra_setlaplacebc (int * matrix_id, int *row, int * numVals, int *columns, double * values);
int msi_epetra_freeze(int* matrix_id); 
int msi_epetra_multiply(int* matrix_id, pField in_field, pField out_field);
int msi_epetra_write(int* matrix_id, const char*, int* skip_zero, int* start_index);
int msi_epetra_print(int* matrix_id);

int msi_solver_aztec(int* matrix_id, pField x_field, pField b_field,
                       int* num_iter, double* tolerance,
		       const char* krylov_solver, const char*
		       preconditioner, const char* sub_dom_solver,
		       int* overlap, int* graph_fill, double*
		       ilu_drop_tol,  double* ilu_fill,
		       double* ilu_omega, int* poly_ord);
  
int msi_solver_amesos(int* matrix_id, pField in_field, pField out_field, const char* solver_name);
int msi_solver_getnumiter(int* matrix_id, int * iter_num);
#endif //#ifdef MSI_TRILINOS
#endif
