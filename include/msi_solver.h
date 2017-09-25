/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifndef MSI_SOLVER_H
#define MSI_SOLVER_H
#include "pumi.h"
#include "msi.h"
#include <map>

// helper routines
int msi_field_getglobaldofid (pField f, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);
int msi_field_getnumowndof (pField f, int* /* out */ num_own_dof);
int msi_field_getdataptr (pField f, double** pts);
int msi_field_getowndofid (pField f, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);

class msi_solver
{
public:
  // functions
  msi_solver();
  ~msi_solver();
  static msi_solver* instance();

  // field 
  int get_num_value(pField);
  void add_field(pField f, int i);
  std::map<pField, int>* field_container;
  pMeshEnt* vertices;

  // adjacency tag
  void set_node_adj_tag();
  pMeshTag num_global_adj_node_tag;
  pMeshTag num_own_adj_node_tag;

  pOwnership ownership;
  pNumbering local_n;
  pNumbering global_n;
private:
  static msi_solver* _instance;
};

#endif
