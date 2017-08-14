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
                         /*2*/ MSI_FIXED};

// START OF API
// remember to delete ownership after use
void msi_start(pMesh m, pOwnership o);
void msi_finalize(pMesh m);

int msi_ent_setdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
                          int* /* out */ num_dof, double* dof_data);
int msi_ent_getdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
                          int* /* out */ num_dof, double* dof_data);

pField msi_field_create (const char* /* in */ field_name, 
                      int /*in*/ num_values, int /*in*/ num_dofs_per_value);

#ifdef MSI_PETSC
/** matrix and solver functions with PETSc */
void msi_matrix_create(int matrix_id, int matrix_type, pField f);
void msi_matrix_freeze(int matrix_id);
void msi_matrix_delete(int matrix_id);

void msi_matrix_insert(int matrix_id, int row, int column, double* val);
void msi_matrix_add(int matrix_id, int row, int column, double* val);
void msi_matrix_addBlock(int matrix_id, int ielm, int rowVarIdx, int columnVarIdx, double* values);

void msi_matrix_setBC(int matrix_id, int row);
void msi_matrix_setLaplaceBC (int matrix_id, int row, int numVals, int* columns, double* values);

void msi_matrix_solve(int matrix_id, pField rhs_sol);
int msi_matrix_getNumIter(int matrix_id);
void msi_matrix_multiply(int matrix_id, pField inputvec, pField outputvec);

// auxiliary
void msi_matrix_flush(int matrix_id);
void msi_matrix_write(int matrix_id, const char* file_name, int start_index);
void msi_matrix_print(int matrix_id);
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
