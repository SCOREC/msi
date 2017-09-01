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
// helper routines
pMeshEnt get_ent(pMesh mesh, int ent_dim, int ent_id);
void msi_mesh_getnumownent (int* /* in*/ ent_dim, int* /* out */ num_ent);

int msi_ent_getownpartid (int* /* in */ ent_dim, int* /* in */ ent_id, 
                            int* /* out */ owning_partid);
int msi_ent_getlocaldofid(int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
                       int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);
int msi_ent_getglobaldofid (int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
         int* /* out */ start_global_dof_id, int* /* out */ end_global_dof_id_plus_one);

int msi_ent_setdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
                          int* /* out */ num_dof, double* dof_data);
int msi_ent_getdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
                          int* /* out */ num_dof, double* dof_data);

int msi_field_getglobaldofid (pField f, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one);

int msi_field_getNumVal(pField f);
int msi_field_getNumDOF(pField f);
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

  // adjacency tag
  void set_node_adj_tag();
  pMeshTag num_global_adj_node_tag;
  pMeshTag num_own_adj_node_tag;

  pNumbering local_n;
  pNumbering global_n;
private:
  static msi_solver* _instance;
};

#endif
