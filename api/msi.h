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
#define MSI_REAL 0
typedef int FieldID;
// end of to-delete
enum msi_matrix_type { /*0*/ MSI_MULTIPLY=0, 
                       /*1*/ MSI_SOLVE}; 
enum msi_matrix_status { /*0*/ MSI_NOT_FIXED=0,
                         /*2*/ MSI_FIXED};

// helper routines
pMeshEnt get_ent(pMesh mesh, int ent_dim, int ent_id);
void msi_mesh_getnumownent (int* /* in*/ ent_dim, int* /* out */ num_ent);
int msi_ent_setdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, FieldID* field_id, 
                          int* /* out */ num_dof, double* dof_data);
int msi_ent_getdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, FieldID* field_id, 
                          int* /* out */ num_dof, double* dof_data);
int msi_ent_getownpartid (int* /* in */ ent_dim, int* /* in */ ent_id, 
                            int* /* out */ owning_partid);
int msi_ent_getlocaldofid(int* /* in */ ent_dim, int* /* in */ ent_id, FieldID* field_id, 
                       int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);
int msi_ent_getglobaldofid (int* /* in */ ent_dim, int* /* in */ ent_id, FieldID* field_id, 
         int* /* out */ start_global_dof_id, int* /* out */ end_global_dof_id_plus_one);
int msi_field_create (FieldID* /*in*/ field_id, const char* /* in */ field_name, int* /*in*/ num_values, 
int* /*in*/ scalar_type, int* /*in*/ num_dofs_per_value);
int msi_field_delete (FieldID* /*in*/ field_id);

int msi_field_getglobaldofid ( FieldID* field_id, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);
void msi_field_getinfo(int* /*in*/ field_id, 
     char* /* out*/ field_name, int* num_values, int* scalar_type, int* total_num_dof);
int msi_field_getnumowndof (FieldID* field_id, int* /* out */ num_own_dof);
int msi_field_getdataptr (FieldID* field_id, double** pts);
int msi_field_getowndofid (FieldID* field_id, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);

// START OF API
// remember to delete ownership 
void msi_start(pMesh m, pOwnership o);
void msi_finalize(pMesh m);

#ifdef MSI_PETSC
/** matrix and solver functions with PETSc */
int msi_matrix_create(int* matrix_id, int* matrix_type, int* scalar_type, FieldID* field_id); //zerosuperlumatrix_
int msi_matrix_freeze(int* matrix_id); //finalizematrix_
int msi_matrix_delete(int* matrix_id); //deletematrix_

int msi_matrix_insert(int* matrix_id, int* row, int* column, int* scalar_type, double* val);
int msi_matrix_add(int* matrix_id, int* row, int* column, int* scalar_type, double* val); //globalinsertval_

int msi_matrix_insertblock(int* matrix_id, int * ielm, int* rowVarIdx, int * columnVarIdx, double * values);
int msi_matrix_setbc(int* matrix_id, int* row);
int msi_matrix_setlaplacebc (int * matrix_id, int *row, int * numVals, int *columns, double * values);

int msi_matrix_solve(int* matrix_id, FieldID* rhs_sol); //solveSysEqu_
int msi_matrix_getiternum(int* matrix_id, int * iter_num);
int msi_matrix_multiply(int* matrix_id, FieldID* inputvecid, FieldID* outputvecid); //matrixvectormult_

// for performance test
int msi_matrix_flush(int* matrix_id);
int msi_matrix_write(int* matrix_id, const char* file_name, int* start_index);
int msi_matrix_print(int* matrix_id);
#endif // #ifdef MSI_PETSC

#ifdef MSI_TRILINOS
//=========================================================================
/** matrix and solver functions with TRILINOS */
//=========================================================================

int msi_epetra_create(int matrix_id, int matrix_type, int field_id);
int msi_epetra_delete(int matrix_id);

int msi_epetra_insert(int matrix_id, int row, int column, double* val);
int msi_epetra_addblock(int matrix_id, int * ielm, int rowVarIdx, int * columnVarIdx, double * values);

int msi_epetra_setbc(int matrix_id, int row);
int msi_epetra_setlaplacebc (int * matrix_id, int *row, int * numVals, int *columns, double * values);
int msi_epetra_freeze(int matrix_id); 
int msi_epetra_multiply(int matrix_id, int in_fieldid, int out_fieldid);
int msi_epetra_write(int matrix_id, const char*, int skip_zero, int start_index);
int msi_epetra_print(int matrix_id);

int msi_solver_aztec(int matrix_id, int x_fieldid, int
		       b_fieldid, int num_iter, double* tolerance,
		       const char* krylov_solver, const char*
		       preconditioner, const char* sub_dom_solver,
		       int overlap, int graph_fill, double*
		       ilu_drop_tol,  double* ilu_fill,
		       double* ilu_omega, int poly_ord);
  
int msi_solver_getnumiter(int matrix_id, int * iter_num);
#endif //#ifdef MSI_TRILINOS
#endif
