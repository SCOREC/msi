/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifdef MSI_PETSC
#ifndef MSI_SOLVER_H
#define MSI_SOLVER_H
//#include "superlu_ddefs.h" // gridinfo_t
#include "apf.h"
#include "petscksp.h"
#include "apfNumbering.h"
#include "msi.h"
#include <vector>

// helper routines
// helper routines
pMeshEnt get_ent(pMesh mesh, int ent_dim, int ent_id);
void msi_mesh_getnumownent (int* /* in*/ ent_dim, int* /* out */ num_ent);

int msi_ent_getownpartid (int* /* in */ ent_dim, int* /* in */ ent_id, 
                            int* /* out */ owning_partid);
int msi_ent_getlocaldofid(int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
                       int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);
int msi_ent_getglobaldofid (int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
         int* /* out */ start_global_dof_id, int* /* out */ end_global_dof_id_plus_one);

int msi_field_getglobaldofid (pField f, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);
void msi_field_getinfo(pField f, 
     char* /* out*/ field_name, int* num_values, int* total_num_dof);
int msi_field_getnumowndof (pField f, int* /* out */ num_own_dof);
int msi_field_getdataptr (pField f, double** pts);
int msi_field_getowndofid (pField f, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);
int copyField2PetscVec(pField f, Vec& petscVec);
int copyPetscVec2Field(Vec& petscVec, pField f);
void printMemStat();

// NOTE: all field realted interaction is done through msi api rather than apf
class msi_matrix
{
public:
  msi_matrix(int i, pField f);
  virtual ~msi_matrix();
  virtual int initialize()=0; // create a matrix and solver object
  int destroy(); // delete a matrix and solver object
  int set_value(int row, int col, int operation, double real_val, double imag_val); //insertion/addition with global numbering
  // values use column-wise, size * size block
  int add_values(int rsize, int * rows, int csize, int * columns, double* values);
  int get_values(std::vector<int>& rows, std::vector<int>& n_columns, std::vector<int>& columns, std::vector<double>& values);
  void set_status(int s) {mat_status=s;}
  int get_status() {return mat_status;}
  int get_scalar_type() {
#ifdef MSI_COMPLEX
  return 1;
#else
  return 0;
#endif
}
  pField get_field() { return field;}
  int write( const char* file_name);
  virtual int get_type() const = 0;
  virtual int assemble() = 0;
  virtual int setupMat() =0;
  virtual int preAllocate() =0;
  virtual int flushAssembly();
  int printInfo();
  // PETSc data structures
  Mat* A;
protected:
  int setupSeqMat();
  int setupParaMat();
  int preAllocateSeqMat();
  int preAllocateParaMat();
  int id;
  int mat_status; 
  pField field; // the field that provide numbering
};

class matrix_mult: public msi_matrix
{
public:
  matrix_mult(int i,pField f): msi_matrix(i,f), localMat(1) { initialize();}
  virtual int initialize();
  void set_mat_local(bool flag) {localMat=flag;}
  int is_mat_local() {return localMat;}
  int multiply(pField in_f, pField out_f);
  virtual int get_type() const { return 0; } //MSI_MULTIPLY; }
  virtual int assemble();
  virtual int setupMat();
  virtual int preAllocate();
private:
  bool localMat;
};

class matrix_solve: public msi_matrix
{
public:
  matrix_solve(int i, pField f);
  virtual int initialize();
  virtual ~matrix_solve();
  int solve(pField f);
  int set_bc( int row);
  int set_row( int row, int numVals, int* colums, double * vals);
  int add_blockvalues( int rbsize, int * rows, int cbsize, int * columns, double* values);
  virtual int get_type() const {return 1; }
  virtual int assemble(); 
  virtual int setupMat();
  virtual int preAllocate();
  int iterNum;
private:  
  int setUpRemoteAStruct();
  int setKspType();
  int kspSet;
  KSP* ksp; 
  Mat remoteA;
  std::set<int> remotePidOwned;
  std::map<int, std::map<int, int> > remoteNodeRow; // <pid, <locnode>, numAdj >
  std::map<int, int> remoteNodeRowSize;
};

class msi_solver
{
public:
// functions
  msi_solver();
  ~msi_solver();
  static msi_solver* instance();
  msi_matrix* get_matrix(int matrix_id);
  void add_matrix(int matrix_id, msi_matrix*);
// data
  std::map<int, msi_matrix*>* matrix_container;
  int assembleOption; // 0 use scorec; 1 use petsc
  void set_node_adj_tag();
  pMeshTag num_global_adj_node_tag;
  pMeshTag num_own_adj_node_tag;
private:
  static msi_solver* _instance;
};

#endif
#endif //#ifndef MSI_MESHGEN
