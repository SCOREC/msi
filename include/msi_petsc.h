/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifdef MSI_PETSC
#ifndef MSI_PETSC_H
#define MSI_PETSC_H
#include "pumi.h"
#include "petscksp.h"
#include "msi.h"
#include "msi_solver.h"
#include <vector>
#include <map>
int copyField2PetscVec(pField f, Vec& petscVec);
int copyPetscVec2Field(Vec& petscVec, pField f);
void printMemStat();

// NOTE: all field realted interaction is done through msi api rather than apf
class msi_matrix
{
public:
  msi_matrix(pField f);
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
  int mat_status; 
  pField field; // the field that provide numbering
};

class matrix_mult: public msi_matrix
{
public:
  matrix_mult(pField f): msi_matrix(f), localMat(1) { initialize();}
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
  matrix_solve(pField f);
  virtual int initialize();
  virtual ~matrix_solve();
  int solve(pField rhs, pField sol);
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
  int setUpRemoteAStructParaMat();
  int setKspType();
  int kspSet;
  KSP* ksp; 
  Mat remoteA;
  std::set<int> remotePidOwned;
  std::map<int, std::map<int, int> > remoteNodeRow; // <pid, <locnode>, numAdj >
  std::map<int, int> remoteNodeRowSize;
};

#endif
#endif //#ifdef MSI_PETSC
