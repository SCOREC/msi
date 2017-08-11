/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifdef MSI_PETSC
#include "msi.h"
#include "msi_petsc.h"
#include "apf.h"
#include "apfNumbering.h"
#include "apfShape.h"
#include "apfMesh.h"
#include <vector>
#include <set>
#include "PCU.h"
#include <assert.h>
#include <iostream>

#ifdef PETSC_USE_COMPLEX
#include "petscsys.h" // for PetscComplex
#include <complex>
using std::complex;
#endif

// ***********************************
// 		MSI_SOLVER
// ***********************************

msi_solver* msi_solver::_instance=NULL;
msi_solver* msi_solver::instance()
{
  if (_instance==NULL)
    _instance = new msi_solver();
  return _instance;
}

struct entMsg
{
  int pid;
  apf::MeshEntity* ent;
  entMsg( int pid_p=0, apf::MeshEntity* ent_p=NULL)
  {
    pid=pid_p;
    ent=ent_p;
  }
};

struct classcomp
{
  bool operator() (const entMsg& lhs, const entMsg& rhs) const
  {
    if(lhs.ent==rhs.ent) return lhs.pid<rhs.pid;
    else return lhs.ent<rhs.ent;
  }
};


// **********************************************
void set_adj_node_tag(pMeshTag num_global_adj_node_tag, pMeshTag num_own_adj_node_tag)
// **********************************************
{
  pMesh mesh = pumi::instance()->mesh;

  int value;
  int brgType = mesh->getDimension()-1;

  apf::MeshEntity* e;
  apf::MeshIterator* it = mesh->begin(0);
  PCU_Comm_Begin();
  while ((e = mesh->iterate(it)))
  {
    int num_adj_node=0;
    apf::Adjacent elements;
    apf::getBridgeAdjacent(mesh, e, brgType, 0, elements);
    int num_adj = elements.getSize();

    for (int i=0; i<num_adj; ++i)
    {
      if (pumi_ment_isOwned(elements[i]))
        ++num_adj_node;
    }
    mesh->setIntTag(e, num_own_adj_node_tag, &num_adj_node);

    if (!mesh->isShared(e)) continue;
    // first pass msg size to owner
    int own_partid = pumi_ment_getOwnPID(e);
    apf::MeshEntity* own_copy = pumi_ment_getOwnEnt(e);

    if (own_partid==PCU_Comm_Self()) continue;
    PCU_COMM_PACK(own_partid, own_copy);
    PCU_Comm_Pack(own_partid, &num_adj,sizeof(int));
  }
  mesh->end(it);

  PCU_Comm_Send();

  std::map<apf::MeshEntity*, std::map<int, int> > count_map;
  while (PCU_Comm_Listen())
  {
    while ( ! PCU_Comm_Unpacked())
    {
      PCU_COMM_UNPACK(e);
      PCU_Comm_Unpack(&value,sizeof(int));
      count_map[e][PCU_Comm_Sender()]=value;
    }
  }
  
  // pass entities to ownner
  std::map<apf::MeshEntity*, std::set<entMsg, classcomp> > count_map2;
  it = mesh->begin(0);
  PCU_Comm_Begin();
  while ((e = mesh->iterate(it)))
  {
    // pass entities to ownner

    std::vector<entMsg> msgs;
    apf::Adjacent elements;
    apf::getBridgeAdjacent(mesh, e, brgType, 0, elements);

    apf::MeshEntity* ownerEnt=pumi_ment_getOwnEnt(e);
    int own_partid = pumi_ment_getOwnPID(e);
    for(int i=0; i<elements.getSize(); ++i)
    {
      apf::MeshEntity* ownerEnt2=pumi_ment_getOwnEnt(elements[i]);
      int owner=pumi_ment_getOwnPID(elements[i]);
      msgs.push_back(entMsg(owner, ownerEnt2));
      if(own_partid==PCU_Comm_Self()) 
      {
        count_map2[e].insert(*msgs.rbegin());
      }
    }

    if(own_partid!=PCU_Comm_Self())
    {
      PCU_COMM_PACK(own_partid, ownerEnt);
      PCU_Comm_Pack(own_partid, &msgs.at(0),sizeof(entMsg)*msgs.size());
    }
  }
  mesh->end(it);
  PCU_Comm_Send();
  while (PCU_Comm_Listen())
  {
    while ( ! PCU_Comm_Unpacked())
    {
      PCU_COMM_UNPACK(e);
      int sizeData = count_map[e][PCU_Comm_Sender()];
      std::vector<entMsg> data(sizeData);
      PCU_Comm_Unpack(&data.at(0),sizeof(entMsg)*sizeData);
      for (int i=0; i<data.size(); ++i)
      {
        count_map2[e].insert(data.at(i));
      }
    }
  }

  for (std::map<apf::MeshEntity*, std::set<entMsg,classcomp> >::iterator mit=count_map2.begin(); 
       mit!=count_map2.end(); ++mit)
  {
    e = mit->first;
    int num_global_adj =count_map2[e].size();
    mesh->setIntTag(mit->first, num_global_adj_node_tag, &num_global_adj);
  }
}


msi_solver::msi_solver()
: assembleOption(0) 
{
  matrix_container = new std::map<int, msi_matrix*>;
  PetscMemorySetGetMaximumUsage();
  num_global_adj_node_tag = pumi::instance()->mesh->createIntTag("msi_num_global_adj_node", 1);
  num_own_adj_node_tag = pumi::instance()->mesh->createIntTag("msi_num_own_adj_node", 1);
  set_adj_node_tag(num_global_adj_node_tag, num_own_adj_node_tag);
}

msi_solver::~msi_solver()
{
  pMesh mesh = pumi::instance()->mesh;
  apf::removeTagFromDimension(mesh, num_global_adj_node_tag, 0);
  mesh->destroyTag(num_global_adj_node_tag);
  apf::removeTagFromDimension(mesh, num_own_adj_node_tag, 0);
  mesh->destroyTag(num_own_adj_node_tag);

  if (matrix_container!=NULL)
    matrix_container->clear();
  matrix_container=NULL;
  delete _instance;
}

void msi_solver::add_matrix(int matrix_id, msi_matrix* matrix)
{
  assert(matrix_container->find(matrix_id)==matrix_container->end());
  matrix_container->insert(std::map<int, msi_matrix*>::value_type(matrix_id, matrix));
}

msi_matrix* msi_solver::get_matrix(int matrix_id)
{
  std::map<int, msi_matrix*>::iterator mit = matrix_container->find(matrix_id);
  if (mit == matrix_container->end()) 
    return (msi_matrix*)NULL;
  return mit->second;
}
#endif
