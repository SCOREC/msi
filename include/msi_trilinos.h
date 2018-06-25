/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifdef MSI_TRILINOS
#ifndef MSI_LS_H
#define MSI_LS_H
#include <map>
#include "apf.h"
#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
typedef int global_ordinal_type;
#else
//typedef long long global_ordinal_type; -- 64bit should throw compilation error
#endif

void write_matrix(Epetra_CrsMatrix*, const char* filename, bool nozero, int start_index);

// NOTE: all field realted interaction is done through msi api rather than apf
class msi_epetra
{
public:
  msi_epetra(int i, int t, int s, FieldID field);
  ~msi_epetra();
  void destroy(); // delete a matrix and solver object
  int get_scalar_type() { return scalar_type; }
  int get_field_id() { return field_id;}
  Epetra_CrsMatrix* epetra_mat;
  Epetra_Map* _overlap_map;
  Epetra_Map* _owned_map;
  global_ordinal_type nge;
  int num_solver_iter;
  int matrix_type; /* 0 for multiplication, 1 for solver*/
protected:
  int id;
  int scalar_type;
  apf::Field* _field;
  int field_id; // the field that provides numbering
};

class msi_ls
{
public:
// functions
  msi_ls(){matrix_container = new std::map<int, msi_epetra*>;}
  ~msi_ls();
  static msi_ls* instance();
  msi_epetra* get_matrix(int matrix_id);
  void add_matrix(int matrix_id, msi_epetra*);
// data
  std::map<int, msi_epetra*>* matrix_container;
private:
  static msi_ls* _instance;
};
#endif
#endif
