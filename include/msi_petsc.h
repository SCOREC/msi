/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifndef MSI_PETSC_HEADER_H
#define MSI_PETSC_HEADER_H
#include "apf.h"
#include "petscksp.h"
#include "pumi.h"
#include <vector>

class msi_matrix
{
public:
  msi_matrix(int i, pField f, int n); // constructor cannot be virtual
  virtual ~msi_matrix(); // the base class destructor automatically executes after the derived-class destructor
  virtual int initialize()=0; // create a matrix and solver object
  int destroy(); // delete a matrix and solver object
  int set_value(int row, int col, int operation, double real_val, double imag_val); //insertion/addition with global numbering
  // values use column-wise, size * size block
  int add_values(int rsize, int * rows, int csize, int * columns, double* values);
  int get_values(std::vector<int>& rows, std::vector<int>& n_columns, std::vector<int>& columns, std::vector<double>& values);
  void set_status(int s) {mat_status=s;}
  int get_status() {return mat_status;}
  int get_scalar_type() { return scalar_type; }
  pField get_field() { return field;}
  int get_num_field_value() { return num_values;}
  int write( const char* file_name);
  virtual int get_type() const = 0;
  virtual int assemble() = 0;
  virtual int setupMat() =0;
  virtual void preAllocate() =0;
  virtual int flushAssembly();
  void printInfo();
  // PETSc data structures
  Mat* A;
protected:
  int setupSeqMat();
  int setupParaMat();
  int preAllocateSeqMat();
  int preAllocateParaMat();
  int id;
  int scalar_type;
  int mat_status; 
  pField field; // the field that provide dof numbering
  int num_values;
};

class matrix_mult: public msi_matrix
{
public:
  matrix_mult(int i, pField field, int n): msi_matrix(i,field, n), localMat(1) { initialize();}
  virtual int initialize();
  void set_mat_local(bool flag) {localMat=flag;}
  int is_mat_local() {return localMat;}
  int multiply(pField in_field, pField out_field);
  virtual int get_type() const { return 0; } //MSI_MULTIPLY; }
  virtual int assemble();
  virtual int setupMat();
  virtual void preAllocate();
private:
  bool localMat;
};

class matrix_solve: public msi_matrix
{
public:
  matrix_solve(int i, pField f, int n);
  virtual int initialize();
  virtual ~matrix_solve();
  int solve(pField x_f, pField b_f);
  void set_bc( int row);
  void set_row( int row, int numVals, int* colums, double * vals);
  void add_blockvalues( int rbsize, int * rows, int cbsize, int * columns, double* values);
  virtual int get_type() const {return 1; }
  virtual int assemble(); 
  virtual int setupMat();
  virtual void preAllocate();
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
  pMeshTag num_global_adj_node_tag;
  pMeshTag num_own_adj_node_tag;
private:
  static msi_solver* _instance;
};

#endif
