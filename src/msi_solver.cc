/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifdef MSI_PETSC
#include "msi.h"
#include "msi_solver.h"
#include "apf.h"
#include "apfMDS.h"
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

msi_solver::msi_solver()
{
  field_container = new std::map<pField, int>;
}

msi_solver::~msi_solver()
{
  if (field_container!=NULL)
    field_container->clear();
  field_container=NULL;
  delete _instance;
}

void msi_solver::add_field(pField f, int n)
{
  assert(field_container->find(f)==field_container->end());
  field_container->insert(std::map<pField, int>::value_type(f,n));
}

int msi_solver::get_num_value(pField f)
{
  std::map<pField, int>::iterator mit = field_container->find(f);
  if (mit == field_container->end()) 
    return 0;
  return mit->second;
}

// helper routines


// internal function
//*******************************************************
int msi_ent_getlocaldofid(int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
                       int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  if (*ent_dim!=0)
    return MSI_FAILURE;

  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);
  assert(e);

  msi_ment_getFieldID(e, f, 0, start_dof_id, end_dof_id_plus_one);
  return MSI_SUCCESS;
}

// internal function
//*******************************************************
int msi_ent_getglobaldofid (int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  if (*ent_dim!=0)
    return MSI_FAILURE;

  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);
  assert(e);

  msi_ment_getGlobalFieldID(e, f, 0, start_dof_id, end_dof_id_plus_one);
  return MSI_SUCCESS;
}


void msi_ment_getGlobalFieldID(pMeshEnt e, pField f, int* start_dof_id, int* end_dof_id_plus_one)
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif
  *start_dof_id = pumi_ment_getGlobalID(e)*num_dof;
  *end_dof_id_plus_one = *start_dof_id +num_dof;
}

// helper routines
#define FIELDVALUELIMIT 1e100
bool value_is_nan(double val)
{
  return val!=val ||fabs(val) >FIELDVALUELIMIT;
}

//*******************************************************
void msi_mesh_getnumownent (int* /* in*/ ent_dim, int* /* out */ num_ent)
//*******************************************************
{
  *num_ent = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, *ent_dim);
}


// *********************************************************
pMeshEnt get_ent(apf::Mesh2* mesh, int ent_dim, int ent_id)
// *********************************************************
{
  return apf::getMdsEntity(mesh, ent_dim, ent_id);
}


//*******************************************************
int msi_ent_setdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
                          int* /* out */ num_dof, double* dof_data)
//*******************************************************
{
  assert(*ent_dim==0);
  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);
  assert(e);

#ifdef DEBUG
  int scalar_type=0;
#ifdef PETSC_USE_COMPLEX
  scalar_type=1;
#endif
  assert(countComponents(f)==*num_dof*(1+scalar_type));
  for(int i=0; i<*num_dof*(1+scalar_type); i++)
    assert(!value_is_nan(dof_data[i]));
#endif
  setComponents(f, e, 0, dof_data);
  return MSI_SUCCESS;
}

//*******************************************************
int msi_ent_getdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, pField f, 
                          int* /* out */ num_dof, double* dof_data)
//*******************************************************
{
  assert(*ent_dim==0);
  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);
  assert(e);

  getComponents(f, e, 0, dof_data);

  *num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  *num_dof/=2;
#endif

#ifdef DEBUG
  int scalar_type=0;
#ifdef PETSC_USE_COMPLEX
  scalar_type=1;
#endif
  for(int i=0; i<*num_dof*(1+scalar_type); i++)
    assert(!value_is_nan(dof_data[i]));
  int start_dof_id,end_dof_id_plus_one;
  msi_ent_getlocaldofid(ent_dim, ent_id, f, &start_dof_id, &end_dof_id_plus_one);
  double* data;
  msi_field_getdataptr(f, &data);
  int start=start_dof_id*(1+scalar_type);
  for( int i=0; i< *num_dof; i++)
    assert(data[start++]==dof_data[i]);
#endif
  return MSI_SUCCESS;
}

//*******************************************************
int msi_ent_getownpartid (int* /* in */ ent_dim, int* /* in */ ent_id, 
                            int* /* out */ owning_partid)
//*******************************************************
{
  apf::MeshEntity* e = getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);
  assert(e);
  *owning_partid = pumi_ment_getOwnPID(e);
  return MSI_SUCCESS;
}


//******************************************************* 
int msi_field_getglobaldofid (pField f, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif

  *start_dof_id=0;
  *end_dof_id_plus_one=*start_dof_id+num_dof*pumi_mesh_getNumGlobalEnt(pumi::instance()->mesh, 0);
  return MSI_SUCCESS;
}

//*******************************************************
int msi_field_getnumowndof (pField f, int* /* out */ num_own_dof)
//*******************************************************
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof /= 2;
#endif
  *num_own_dof = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0)*num_dof;
  return MSI_SUCCESS;
}

//*******************************************************
int msi_field_getdataptr (pField f, double** pts)
//*******************************************************
{
  if (!isFrozen(f)) freeze(f);
  *pts=getArrayData(f);
  return MSI_SUCCESS;
}

//*******************************************************
int msi_field_getowndofid (pField f, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  int num_own_ent = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof /=2;
#endif
  
  int start_id = num_own_ent;
  PCU_Exscan_Ints(&start_id,1);

  *start_dof_id=start_id*num_dof;
  *end_dof_id_plus_one=*start_dof_id+num_own_ent*num_dof;
  return MSI_SUCCESS;
}

#endif
