/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#include "msi.h"
#include "msi_petsc.h"
#include <iostream>
#include "PCU.h"
#include "apfMDS.h"
#include <vector>
#include <assert.h>

using std::vector;

extern pOwnership msi_ownership=NULL;

// Error if serial mesh is loaded and partitioned as pumi_ment_getID(e) is not sequential
void msi_ment_getLocalFieldID(pMeshEnt e, pField f, int* start_dof_id, int* end_dof_id_plus_one)
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif
  *start_dof_id = pumi_ment_getID(e)*num_dof;
  *end_dof_id_plus_one = *start_dof_id + num_dof;
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

int msi_field_getNumOwnDOF(pField f)
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif
  return num_dof*pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
}

void msi_field_getOwnDOFID(pField f, 
    int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif
  int num_own_ent = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
  int start_id = num_own_ent;
  PCU_Exscan_Ints(&start_id,1);

  *start_dof_id=start_id*num_dof;
  *end_dof_id_plus_one=*start_dof_id+num_own_ent*num_dof;
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
int msi_ent_setdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, FieldID* field_id, 
                          int* /* out */ num_dof, double* dof_data)
//*******************************************************
{
  assert(*ent_dim==0);
  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);
  assert(e);

  apf::Field* f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);

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
int msi_ent_getdofdata (int* /* in */ ent_dim, int* /* in */ ent_id, FieldID* field_id, 
                          int* /* out */ num_dof, double* dof_data)
//*******************************************************
{
  assert(*ent_dim==0);
  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);
  assert(e);

  apf::Field* f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);

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
  msi_ent_getlocaldofid(ent_dim, ent_id,field_id, &start_dof_id, &end_dof_id_plus_one);
  double* data;
  msi_field_getdataptr(field_id, &data);
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
int msi_ent_getlocaldofid(int* /* in */ ent_dim, int* /* in */ ent_id, FieldID* field_id, 
                       int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  if (*ent_dim!=0)
    return MSI_FAILURE;

  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);
  assert(e);

  apf::Field* f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif
  *start_dof_id = *ent_id*num_dof;
  *end_dof_id_plus_one = *start_dof_id +num_dof;
  return MSI_SUCCESS;
}



//*******************************************************
int msi_ent_getglobaldofid (int* /* in */ ent_dim, int* /* in */ ent_id, FieldID* field_id, 
         int* /* out */ start_global_dof_id, int* /* out */ end_global_dof_id_plus_one)
//*******************************************************
{
  if (*ent_dim!=0)
    return MSI_FAILURE;

  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);
  assert(e);

  apf::Field* f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);

  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif

  int global_id = pumi_ment_getGlobalID(e);
  *start_global_dof_id = global_id*num_dof;
  *end_global_dof_id_plus_one =*start_global_dof_id + num_dof;
  return MSI_SUCCESS;
}

//******************************************************* 
int msi_field_getglobaldofid ( FieldID* field_id, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  apf::Field* f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif

  *start_dof_id=0;
  *end_dof_id_plus_one=*start_dof_id+num_dof*pumi_mesh_getNumGlobalEnt(pumi::instance()->mesh, 0);
  return MSI_SUCCESS;
}

//*******************************************************
int msi_field_getnumowndof (FieldID* field_id, int* /* out */ num_own_dof)
//*******************************************************
{
  apf::Field* f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof /= 2;
#endif
  *num_own_dof = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0)*num_dof;
  return MSI_SUCCESS;
}

//*******************************************************
int msi_field_getdataptr (FieldID* field_id, double** pts)
//*******************************************************
{
  apf::Field* f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);
  if (!isFrozen(f)) freeze(f);
  *pts=getArrayData(f);
  return MSI_SUCCESS;
}

//*******************************************************
int msi_field_getowndofid (FieldID* field_id, 
         int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  apf::Field* f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);

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
 
//*******************************************************
void msi_field_getinfo(int* /*in*/ field_id, 
                        char* /* out*/ field_name, int* num_values, 
                        int* scalar_type, int* total_num_dof)
//*******************************************************
{
  apf::Field* f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);
  strcpy(field_name, getName(f));
  *num_values = 1;

  *scalar_type = 0;
  *total_num_dof = countComponents(f);
#ifdef PETSC_USE_COMPLEX
  *scalar_type = 1;
  *total_num_dof/=2;
#endif
}

//*******************************************************
void msi_ent_getadj (int* /* in */ ent_dim, int* /* in */ ent_id, 
                      int* /* in */ adj_dim, int* /* out */ adj_ent, 
                      int* /* in */ adj_ent_allocated_size, int* /* out */ num_adj_ent)
//*******************************************************
{
  apf::MeshEntity* e = apf::getMdsEntity(pumi::instance()->mesh, *ent_dim, *ent_id);

  if (*adj_dim>*ent_dim) // upward
  {
    apf::Adjacent adjacent;
    pumi::instance()->mesh->getAdjacent(e,*adj_dim,adjacent);
    *num_adj_ent = adjacent.getSize();
    if (*adj_ent_allocated_size<*num_adj_ent)
      return;
    for (int i=0; i<*num_adj_ent; ++i)
      adj_ent[i] = getMdsIndex(pumi::instance()->mesh, adjacent[i]);
  }
  else if (*adj_dim<*ent_dim) 
  {
    apf::Downward downward;
    *num_adj_ent = pumi::instance()->mesh->getDownward(e, *adj_dim, downward);
    if (*adj_ent_allocated_size<*num_adj_ent)
      return;
    for (int i=0; i<*num_adj_ent; ++i)
      adj_ent[i] = getMdsIndex(pumi::instance()->mesh, downward[i]);
    //adjust the order to work with msi
    if (pumi::instance()->mesh->getDimension()==3 && *ent_dim==3 &&*adj_dim==0 &&adj_ent[0]>adj_ent[3])
    {
      int buff[3];
      memcpy(buff, adj_ent, 3*sizeof(int));
      memcpy(adj_ent, adj_ent+3, 3*sizeof(int));
      memcpy(adj_ent+3, buff, 3*sizeof(int));
    }
  }
}

void msi_start(pMesh m, pOwnership o)
{  
  if (!o && !pumi_rank())
    std::cout<<"[MSI INFO] "<<__func__<<": the default mesh ownership is in use\n";

  msi_ownership=o;
  pumi_mesh_setCount(m, o);
  pumi_mesh_createGlobalID(m, o);
}

void msi_finalize(pMesh m)
{  
  typedef std::map<int, msi_matrix*> matrix_container_map;
  for (std::map<int, msi_matrix*>::iterator it=msi_solver::instance()->matrix_container->begin();
      it!=msi_solver::instance()->matrix_container->end(); ++it)
  {
    if(!PCU_Comm_Self()) std::cout<<"[MSI INFO] "<<__func__<<": matrix "<<it->first<<" deleted\n";
    delete it->second;
  }
  msi_solver::instance()->matrix_container->clear();

  while(m->countFields())
  {
    apf::Field* f = m->getField(0);
    if(!PCU_Comm_Self()) std::cout<<"[MSI INFO] "<<__func__<<": field "<<getName(f)<<" deleted\n";
    destroyField(f);
  }

  pumi_mesh_deleteGlobalID(m);  // delete global id
}



//*******************************************************
void msi_field_assign(int* field_id, double* fac, int scalar_type)
//*******************************************************
{
  pField f = pumi_mesh_getField(pumi::instance()->mesh, *field_id);
  int dofPerEnt = countComponents(f);
#ifdef PETSC_USE_COMPLEX
  dofPerEnt /= 2;
#endif

  int vertex_type=0, num_vtx=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  std::vector<double> dofs(dofPerEnt*(1+scalar_type), fac[0]);
  if(scalar_type)
    for(int i=0; i<dofPerEnt; i++)
      dofs.at(2*i+1)=fac[1];
  for(int inode=0; inode<num_vtx; inode++)
  {
    msi_ent_setdofdata (&vertex_type, &inode, field_id, &dofPerEnt, &dofs[0]);
  }
}


int msi_field_create (FieldID* /*in*/ field_id, const char* /* in */ field_name, int* /*in*/ num_values, 
int* /*in*/ scalar_type, int* /*in*/ num_dofs_per_value)
{
  int components = (*num_values)*(*scalar_type+1)*(*num_dofs_per_value);
  apf::Field* f = createPackedField(pumi::instance()->mesh, field_name, components);
  apf::freeze(f); // switch dof data from tag to array
  double val[2]={0,0};
  msi_field_assign(field_id, val, *scalar_type);
  return MSI_SUCCESS;
}

//*******************************************************
int msi_field_delete (FieldID* /*in*/ field_id)
//*******************************************************
{
  pMesh m = pumi::instance()->mesh;
  apf::Field* f = pumi_mesh_getField(m, *field_id);
  destroyField(f);
  return MSI_SUCCESS;
}

#ifdef MSI_PETSC
/** matrix and solver functions */
std::map<int, int> matHit;
int getMatHit(int id) { return matHit[id];};
void addMatHit(int id) { matHit[id]++; }

//*******************************************************
int msi_matrix_create(int* matrix_id, int* matrix_type, int* scalar_type, FieldID *field_id)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);

  if (mat)
  {
    if (!PCU_Comm_Self())
      std::cout <<"[M3D-C1 ERROR] "<<__func__<<" failed: matrix with id "<<*matrix_id<<" already created\n";
    return MSI_FAILURE; 
  }

  if (*matrix_type==MSI_MULTIPLY) // matrix for multiplication
  {
    matrix_mult* new_mat = new matrix_mult(*matrix_id, *scalar_type, *field_id);
    msi_solver::instance()->add_matrix(*matrix_id, (msi_matrix*)new_mat);
  }
  else 
  {
    matrix_solve* new_mat= new matrix_solve(*matrix_id, *scalar_type, *field_id);
    msi_solver::instance()->add_matrix(*matrix_id, (msi_matrix*)new_mat);
  }

#ifdef DEBUG
  if (!PCU_Comm_Self())
    std::cout<<"[M3D-C1 INFO] "<<__func__<<": ID "<<*matrix_id<<", field "<<*field_id<<"\n";
#endif 

  return MSI_SUCCESS;
}

//*******************************************************
int msi_matrix_freeze(int* matrix_id) 
//*******************************************************
{
  double t1 = MPI_Wtime();
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;
  mat->assemble();
  return MSI_SUCCESS;
}

//*******************************************************
int msi_matrix_delete(int* matrix_id)
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;

#ifdef DEBUG
  if (!PCU_Comm_Self())
    std::cout<<"[M3D-C1 INFO] "<<__func__<<": ID "<<*matrix_id<<"\n";
#endif

  typedef std::map<int, msi_matrix*> matrix_container_map;
  msi_solver::instance()->matrix_container->erase(matrix_container_map::key_type(*matrix_id));
  delete mat;
  return MSI_SUCCESS;
}

//*******************************************************
int msi_matrix_insert(int* matrix_id, int* row, 
         int* col, int* scalar_type, double* val)
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;
  if (mat->get_status()==MSI_FIXED)
  {
    if (!PCU_Comm_Self())
      std::cout <<"[M3D-C1 ERROR] "<<__func__<<" failed: matrix with id "<<*matrix_id<<" is fixed\n";
    return MSI_FAILURE;
  }

#ifdef DEBUG
  int field = mat->get_fieldOrdering();
  int num_values, value_type, total_num_dof;
  char field_name[256];
  msi_field_getinfo(&field, field_name, &num_values, &value_type, &total_num_dof);

  int ent_id = *row/total_num_dof;
  apf::MeshEntity* e =apf::getMdsEntity(pumi::instance()->mesh, 0, ent_id);
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));
#endif

  if (*scalar_type==1)
    mat->set_value(*row, *col, INSERT_VALUES, val[0], val[1]);
  else
    mat->set_value(*row, *col, INSERT_VALUES, *val, 0);
  return MSI_SUCCESS;
}

//*******************************************************
int msi_matrix_add (int* matrix_id, int* row, int* col, 
                      int* scalar_type, double* val) //globalinsertval_
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;
  if (mat->get_status()==MSI_FIXED)
  {
    if (!PCU_Comm_Self())
      std::cout <<"[M3D-C1 ERROR] "<<__func__<<" failed: matrix with id "<<*matrix_id<<" is fixed\n";
    return MSI_FAILURE;
  }

#ifdef DEBUG
  int field = mat->get_fieldOrdering();
  int num_values, value_type, total_num_dof;
  char field_name[256];
  msi_field_getinfo(&field, field_name, &num_values, &value_type, &total_num_dof);

  int ent_id = *row/total_num_dof;
  apf::MeshEntity* e =apf::getMdsEntity(pumi::instance()->mesh, 0, ent_id);
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));
#endif

  if (*scalar_type==1)
    mat->set_value(*row, *col, ADD_VALUES, val[0], val[1]);
  else
    mat->set_value(*row, *col, ADD_VALUES, *val, 0);
  return MSI_SUCCESS;
}

//*******************************************************
int msi_matrix_setbc(int* matrix_id, int* row)
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;

  if (mat->get_type()!=MSI_SOLVE)
  { 
    if (!PCU_Comm_Self())
      std::cout <<"[M3D-C1 ERROR] "<<__func__<<" not supported with matrix for multiplication (id"<<*matrix_id<<")\n";
    return MSI_FAILURE;
  }
  int field = mat->get_fieldOrdering();
  int num_values, value_type, total_num_dof;
  char field_name[256];
  msi_field_getinfo(&field, field_name, &num_values, &value_type, &total_num_dof);
  int inode = *row/total_num_dof;
  int ent_dim=0, start_global_dof_id, end_global_dof_id_plus_one;
  msi_ent_getglobaldofid (&ent_dim, &inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);

#ifdef DEBUG
  apf::MeshEntity* e =apf::getMdsEntity(pumi::instance()->mesh, 0, inode);
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));

  int start_dof_id, end_dof_id_plus_one;
  msi_ent_getlocaldofid (&ent_dim, &inode, &field, &start_dof_id, &end_dof_id_plus_one);
  assert(*row>=start_dof_id&&*row<end_dof_id_plus_one);
#endif
  int row_g = start_global_dof_id+*row%total_num_dof;
  (dynamic_cast<matrix_solve*>(mat))->set_bc(row_g);
}

//*******************************************************
int msi_matrix_setlaplacebc(int * matrix_id, int *row,
         int * numVals, int *columns, double * values)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;

  if (mat->get_type()!=MSI_SOLVE)
  {
    if (!PCU_Comm_Self())
      std::cout <<"[M3D-C1 ERROR] "<<__func__<<" not supported with matrix for multiplication (id"<<*matrix_id<<")\n";
    return MSI_FAILURE;
  }
  std::vector <int> columns_g(*numVals);
  int field = mat->get_fieldOrdering();
  int num_values, value_type, total_num_dof;
  char field_name[256];
  msi_field_getinfo(&field, field_name, &num_values, &value_type, &total_num_dof);
  int inode = *row/total_num_dof;
  int ent_dim=0, start_global_dof_id, end_global_dof_id_plus_one;
  msi_ent_getglobaldofid (&ent_dim, &inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);

#ifdef DEBUG
  apf::MeshEntity* e =apf::getMdsEntity(pumi::instance()->mesh, 0, inode);
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));
  int start_dof_id, end_dof_id_plus_one;
  msi_ent_getlocaldofid (&ent_dim, &inode, &field, &start_dof_id, &end_dof_id_plus_one);
  assert(*row>=start_dof_id&&*row<end_dof_id_plus_one);
  for (int i=0; i<*numVals; i++)
    assert(columns[i]>=start_dof_id&&columns[i]<end_dof_id_plus_one);
#endif

  int row_g = start_global_dof_id+*row%total_num_dof;
  for(int i=0; i<*numVals; i++)
  {
    columns_g.at(i) = start_global_dof_id+columns[i]%total_num_dof;
  }
  (dynamic_cast<matrix_solve*>(mat))->set_row(row_g, *numVals, &columns_g[0], values);
}

int msi_matrix_solve(int* matrix_id, FieldID* rhs_sol) //solveSysEqu_
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;
  if (mat->get_type()!=MSI_SOLVE)
  { 
    if (!PCU_Comm_Self())
      std::cout <<"[M3D-C1 ERROR] "<<__func__<<" not supported with matrix for multiplication (id"<<*matrix_id<<")\n";
    return MSI_FAILURE;
  }
  if (!PCU_Comm_Self())
     std::cout <<"[M3D-C1 INFO] "<<__func__<<": matrix "<<* matrix_id<<", field "<<*rhs_sol<<"\n";

  (dynamic_cast<matrix_solve*>(mat))->solve(*rhs_sol);

  addMatHit(*matrix_id);
}

//*******************************************************
int msi_matrix_multiply(int* matrix_id, FieldID* inputvecid, 
         FieldID* outputvecid) 
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;

  if (mat->get_type()!=MSI_MULTIPLY)
  { 
    if (!PCU_Comm_Self())
      std::cout <<"[M3D-C1 ERROR] "<<__func__<<" not supported with matrix for solving (id"<<*matrix_id<<")\n";
    return MSI_FAILURE;
  }

  (dynamic_cast<matrix_mult*>(mat))->multiply(*inputvecid, *outputvecid);
  addMatHit(*matrix_id);
}

//*******************************************************
int msi_matrix_flush(int* matrix_id)
//*******************************************************
{
  double t1 = MPI_Wtime();
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;
  mat->flushAssembly();
}

//*******************************************************
int msi_matrix_getiternum(int* matrix_id, int * iter_num)
//*******************************************************
{ 
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;
  *iter_num = dynamic_cast<matrix_solve*> (mat)->iterNum;
}

//*******************************************************
int msi_matrix_insertblock(int* matrix_id, int * ielm, 
          int* rowIdx, int * columnIdx, double * values)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;
  int field = mat->get_fieldOrdering();
  // need to change later, should get the value from field calls ...
  int dofPerVar = 6;
  char field_name[256];
  int num_values, value_type, total_num_dof; 
  msi_field_getinfo(&field, field_name, &num_values, &value_type, &total_num_dof);
  dofPerVar=total_num_dof/num_values;
  int nodes[6];
  int ent_dim=0;
  int ielm_dim = 2;
  int nodes_per_element=sizeof(nodes)/sizeof(int), nodes_per_element_get;

  if (pumi::instance()->mesh->getDimension()==3) ielm_dim =3;

  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, ielm_dim, *ielm);
  assert(e);
  if (pumi::instance()->mesh->isGhost(e)) return MSI_FAILURE;
  
  msi_ent_getadj (&ielm_dim, ielm, &ent_dim, nodes, &nodes_per_element, &nodes_per_element_get);
  nodes_per_element=nodes_per_element_get;
  int start_global_dof_id,end_global_dof_id_plus_one;
  int start_global_dof,end_global_dof_id;
  // need to change later, should get the value from field calls ...
  int scalar_type = mat->get_scalar_type();
  assert(scalar_type==value_type);
  int numDofs = total_num_dof;
  int numVar = numDofs/dofPerVar;
  assert(*rowIdx<numVar && *columnIdx<numVar);
  int rows[1024], columns[1024];
  assert(sizeof(rows)/sizeof(int)>=dofPerVar*nodes_per_element);
  if(mat->get_type()==0)
  {
    int localFlag=0;
    matrix_mult* mmat = dynamic_cast<matrix_mult*> (mat);
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      if(mmat->is_mat_local()) msi_ent_getlocaldofid (&ent_dim, nodes+inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);
      else msi_ent_getglobaldofid (&ent_dim, nodes+inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+(*rowIdx)*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+(*columnIdx)*dofPerVar+i;
      }
    }
    mmat->add_values(dofPerVar*nodes_per_element, rows,dofPerVar*nodes_per_element, columns, values);
  }
  else
  {
    matrix_solve* smat = dynamic_cast<matrix_solve*> (mat);
    int nodeOwner[6];
    int columns_bloc[6], rows_bloc[6];
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      msi_ent_getownpartid (&ent_dim, nodes+inode, nodeOwner+inode);
      msi_ent_getglobaldofid (&ent_dim, nodes+inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);
      rows_bloc[inode]=nodes[inode]*numVar+*rowIdx;
      columns_bloc[inode]=nodes[inode]*numVar+*columnIdx;
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+(*rowIdx)*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+(*columnIdx)*dofPerVar+i;
      }
    }
    int numValuesNode = dofPerVar*dofPerVar*nodes_per_element*(1+scalar_type);
    int offset=0;
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      if(nodeOwner[inode]!=PCU_Comm_Self()&&!msi_solver::instance()->assembleOption)
        smat->add_blockvalues(1, rows_bloc+inode, nodes_per_element, columns_bloc, values+offset);
      else 
        smat->add_values(dofPerVar, rows+dofPerVar*inode, dofPerVar*nodes_per_element, columns, values+offset);
      offset+=numValuesNode;
    }
  }
}

//*******************************************************
int msi_matrix_write(int* matrix_id, const char* filename, int* start_index)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;
  if (!filename) 
    return msi_matrix_print(matrix_id);

  char matrix_filename[256];
  sprintf(matrix_filename,"%s-%d",filename, PCU_Comm_Self());
  FILE * fp =fopen(matrix_filename, "w");

  int row, col, csize, sum_csize=0, index=0;

  vector<int> rows;
  vector<int> n_cols;
  vector<int> cols;
  vector<double> vals;

  mat->get_values(rows, n_cols, cols, vals);
  for (int i=0; i<rows.size(); ++i)
    sum_csize += n_cols[i];
  assert(vals.size()==sum_csize);

  fprintf(fp, "%d\t%d\t%d\n", rows.size(), n_cols.size(), vals.size());

  for (int i=0; i<rows.size(); ++i)
  {
    row = rows[i];
    csize = n_cols[i];
    for (int j=0; j<csize; ++j)
    {
      fprintf(fp, "%d\t%d\t%E\n", row+*start_index, cols[index]+*start_index,vals[index]);
      ++index;
    }
  }
  fclose(fp);  
  assert(index == vals.size());
  return MSI_SUCCESS;
}


//*******************************************************
int msi_matrix_print(int* matrix_id)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(*matrix_id);
  if (!mat)
    return MSI_FAILURE;

  int row, col, csize, sum_csize=0, index=0;

  vector<int> rows;
  vector<int> n_cols;
  vector<int> cols;
  vector<double> vals;

  mat->get_values(rows, n_cols, cols, vals);
  for (int i=0; i<rows.size(); ++i)
    sum_csize += n_cols[i];
  assert(vals.size()==sum_csize);

  if (!PCU_Comm_Self()) 
    std::cout<<"[M3D-C1 INFO] "<<__func__<<": printing matrix "<<*matrix_id<<"\n";

  for (int i=0; i<rows.size(); ++i)
  {
    row = rows[i];
    csize = n_cols[i];
    for (int j=0; j<csize; ++j)
    {
      std::cout<<"["<<PCU_Comm_Self()<<"]\t"<<row<<"\t"<<cols[index]<<"\t"<<vals[index]<<"\n";
      ++index;
    }
  }
  assert(index == vals.size());
  return MSI_SUCCESS;
}

#endif // #ifdef MSI_PETSC


#ifdef MSI_TRILINOS
#include <Epetra_MultiVector.h>
#include <AztecOO.h>
#include <Epetra_Version.h>
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include "msi_trilinos.h"

void verifyFieldEpetraVector(apf::Field* f, Epetra_MultiVector* x)
{
  double* field_data =getArrayData(f);
  assert(countComponents(f)*m3dc1_mesh::instance()->mesh->count(0)==x->MyLength());

  for(int i=0; i<x->MyLength(); ++i)
  { 
    assert(!value_is_nan((*x)[0][i]) && !value_is_nan(field_data[i]));

    if (!(m3dc1_double_isequal((*x)[0][i], field_data[i])))
      std::cout<<"[p"<<PCU_Comm_Self()<<"] x["<<i<<"]="<<(*x)[0][i]
                <<", field_data["<<i<<"]="<<field_data[i]<<"\n";
      assert(m3dc1_double_isequal((*x)[0][i], field_data[i]));
  }
}

int m3dc1_epetra_create(int* matrix_id, int* matrix_type, int* scalar_type, FieldID* field_id)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);

  if (mat)
  {
    if (!PCU_Comm_Self())
      std::cout <<"[M3D-C1 ERROR] "<<__func__<<" failed: matrix with id "<<*matrix_id<<" already created\n";
    return M3DC1_FAILURE; 
  }
  // check field exists
  if (!m3dc1_mesh::instance()->field_container || !m3dc1_mesh::instance()->field_container->count(*field_id))
  {
    if (!PCU_Comm_Self())
      std::cout <<"[M3D-C1 ERROR] "<<__func__<<" failed: field with id "<<*field_id<<" doesn't exist\n";
    return M3DC1_FAILURE; 
  }
  m3dc1_ls::instance()->add_matrix(*matrix_id, new m3dc1_epetra(*matrix_id, *matrix_type, *scalar_type, *field_id));
  return M3DC1_SUCCESS;
}

int m3dc1_epetra_delete(int* matrix_id)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat)
    return M3DC1_FAILURE;

  typedef std::map<int, m3dc1_epetra*> matrix_container_map;
  m3dc1_ls::instance()->matrix_container->erase(matrix_container_map::key_type(*matrix_id));
  mat->destroy();
  delete mat;
  return M3DC1_SUCCESS;
}

int m3dc1_epetra_insert(int* matrix_id, int* row, int* col, int* scalar_type, double* val)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat)
    return M3DC1_FAILURE;
  assert(*scalar_type==M3DC1_REAL);

  int err = mat->epetra_mat->ReplaceGlobalValues(*row, 1, val, col);
  if (err) {
    err =mat->epetra_mat->InsertGlobalValues(*row, 1, val, col);
    assert(err == 0);
  }
  return M3DC1_SUCCESS;
}

void print_elem (int elem_id)
{
  int ielm_dim;
  apf::Mesh2* m=m3dc1_mesh::instance()->mesh;
  
  int num_node_per_element;
  apf::Downward downward;

  ielm_dim = (m->getDimension()==2)? 2:3; 
  apf::MeshEntity* e = getMdsEntity(m, ielm_dim, elem_id);
  num_node_per_element = m->getDownward(e, 0, downward);
 
  int *id = new int[num_node_per_element];
  for (int i=0; i<num_node_per_element; ++i)
    id[i] = get_ent_globalid(m, downward[i]);

  switch (num_node_per_element)
  { 
    case 3: std::cout <<"["<<PCU_Comm_Self()<<"] elem "<<elem_id<<": nodes "
                      <<id[0]<<" "<<id[1]<<" "<<id[2]<<"\n";
            break;
    case 4: std::cout <<"["<<PCU_Comm_Self()<<"] elem "<<elem_id<<": nodes "
                      <<id[0]<<" "<<id[1]<<" "<<id[2]<<" "<<id[3]<<"\n";
            break;
    case 5: std::cout <<"["<<PCU_Comm_Self()<<"] elem "<<elem_id<<": nodes "
                      <<id[0]<<" "<<id[1]<<" "<<id[2]<<" "<<id[3]<<" "<<id[4]<<"\n";
            break;
    case 6: std::cout <<"["<<PCU_Comm_Self()<<"] elem "<<elem_id<<": nodes "
                      <<id[0]<<" "<<id[1]<<" "<<id[2]<<" "<<id[3]<<" "<<id[4]<<" "<<id[5]<<"\n";
            break;
    default: break;
  }
  delete [] id;
}

// equivalent to Petsc::MatSetValues(*A, rsize, rows, csize, columns, &petscValues[0], ADD_VALUES);
void epetra_add_values(Epetra_CrsMatrix* mat, int rsize, int * rows, int csize, int * columns, double* values)
{
  double val[1];
  int col[1];
  assert(!mat->IndicesAreLocal() && !mat->IndicesAreContiguous());

  for (int i=0; i<rsize; i++)
  {
    for(int j=0; j<csize; j++)
    {
      col[0] = columns[j];
      val[0] = values[i*csize+j];
      int ierr = mat->SumIntoGlobalValues(rows[i], 1, val, col);
      if (ierr) 
        ierr =mat->InsertGlobalValues(rows[i], 1, val, col);
      assert(!ierr);
    }
  } // for i
}

int m3dc1_epetra_addblock(int* matrix_id, int * ielm, int* rowVarIdx, int * columnVarIdx, double * values)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);

  if (!mat)
    return M3DC1_FAILURE;

  int field = mat->get_field_id();
  // need to change later, should get the value from field calls ...
  int dofPerVar = 6;
  char field_name[256];
  int num_values, value_type, total_num_dof; 
  m3dc1_field_getinfo(&field, field_name, &num_values, &value_type, &total_num_dof);
  dofPerVar=total_num_dof/num_values;
  int nodes[6];
  int ent_dim=0;
  int ielm_dim = 2;
  int nodes_per_element=sizeof(nodes)/sizeof(int), nodes_per_element_get;

  if (m3dc1_mesh::instance()->mesh->getDimension()==3) ielm_dim =3;
  m3dc1_ent_getadj (&ielm_dim, ielm, &ent_dim, nodes, &nodes_per_element, &nodes_per_element_get);
  nodes_per_element=nodes_per_element_get;
  int start_global_dof_id,end_global_dof_id_plus_one;
  int start_global_dof,end_global_dof_id;
  // need to change later, should get the value from field calls ...
  int scalar_type = mat->get_scalar_type();
  assert(scalar_type==value_type);
  int numDofs = total_num_dof;
  int numVar = numDofs/dofPerVar;
  assert(*rowVarIdx<numVar && *columnVarIdx<numVar);
  int* rows = new int[dofPerVar*nodes_per_element];
  int* columns = new int[dofPerVar*nodes_per_element];

  if (mat->matrix_type==M3DC1_MULTIPLY)
  {
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      m3dc1_ent_getglobaldofid (&ent_dim, nodes+inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+(*rowVarIdx)*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+(*columnVarIdx)*dofPerVar+i;
      }
    }
    //FIXME: mmat->add_values(dofPerVar*nodes_per_element, rows,dofPerVar*nodes_per_element, columns, values);
    epetra_add_values(mat->epetra_mat, dofPerVar*nodes_per_element, 
                      rows,dofPerVar*nodes_per_element, columns, values);     
  }
  else //M3DC1_SOLVE
  {
    int nodeOwner[6];
    int columns_bloc[6], rows_bloc[6];
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      m3dc1_ent_getownpartid (&ent_dim, nodes+inode, nodeOwner+inode);
      m3dc1_ent_getglobaldofid (&ent_dim, nodes+inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);
      rows_bloc[inode]=nodes[inode]*numVar+*rowVarIdx;
      columns_bloc[inode]=nodes[inode]*numVar+*columnVarIdx;
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+(*rowVarIdx)*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+(*columnVarIdx)*dofPerVar+i;
      }
    }
    int numValuesNode = dofPerVar*dofPerVar*nodes_per_element*(1+scalar_type);
    int offset=0;
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      // FIXME: smat->add_values(dofPerVar, rows+dofPerVar*inode, dofPerVar*nodes_per_element, columns, values+offset);
      epetra_add_values(mat->epetra_mat, dofPerVar, rows+dofPerVar*inode, 
                       dofPerVar*nodes_per_element, columns, values+offset);
      offset += numValuesNode;
    }
  }
  delete [] rows;
  delete [] columns;

  return M3DC1_SUCCESS;
}


int m3dc1_epetra_setbc(int* matrix_id, int* row)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat || mat->matrix_type!=M3DC1_SOLVE)
  {
    std::cout <<"[M3D-C1 ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
    return M3DC1_FAILURE;
  }

  int field = mat->get_field_id();
  int num_values, value_type, total_num_dof;
  char field_name[256];
  m3dc1_field_getinfo(&field, field_name, &num_values, &value_type, &total_num_dof);
  int inode = *row/total_num_dof;
  int ent_dim=0, start_global_dof_id, end_global_dof_id_plus_one;
  m3dc1_ent_getglobaldofid (&ent_dim, &inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);
#ifdef DEBUG
  int start_dof_id, end_dof_id_plus_one;
  m3dc1_ent_getlocaldofid (&ent_dim, &inode, &field, &start_dof_id, &end_dof_id_plus_one);
  assert(*row>=start_dof_id&&*row<end_dof_id_plus_one);
#endif
  global_ordinal_type row_g = start_global_dof_id+*row%total_num_dof;
  global_ordinal_type col[1]; col[0] = row_g;
  double val[1]; val[0]=1.0; 
 
  // MatSetValue(*A, row, row, 1.0, ADD_VALUES);

  int err = mat->epetra_mat->SumIntoGlobalValues(row_g, 1, val, col);
  if (err) 
    err =mat->epetra_mat->InsertGlobalValues(row_g, 1, val, col);
  assert(err == 0);

  return M3DC1_SUCCESS;
}

int m3dc1_epetra_setlaplacebc (int * matrix_id, int *row, int * numVals, int *columns, double * values)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat || mat->matrix_type!=M3DC1_SOLVE)
  {
    std::cout <<"[M3D-C1 ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
    return M3DC1_FAILURE;
  }

  std::vector <global_ordinal_type> columns_g(*numVals);
  int field = mat->get_field_id();
  int num_values, value_type, total_num_dof;
  char field_name[256];
  m3dc1_field_getinfo(&field, field_name, &num_values, &value_type, &total_num_dof);
  int inode = *row/total_num_dof;
  int ent_dim=0, start_global_dof_id, end_global_dof_id_plus_one;
  m3dc1_ent_getglobaldofid (&ent_dim, &inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);
#ifdef DEBUG
  int start_dof_id, end_dof_id_plus_one;
  m3dc1_ent_getlocaldofid (&ent_dim, &inode, &field, &start_dof_id, &end_dof_id_plus_one);
  assert(*row>=start_dof_id&&*row<end_dof_id_plus_one);
  for (int i=0; i<*numVals; i++)
    assert(columns[i]>=start_dof_id&&columns[i]<end_dof_id_plus_one);
#endif
  global_ordinal_type row_g = start_global_dof_id+*row%total_num_dof;
  for(int i=0; i<*numVals; i++)
    columns_g.at(i) = start_global_dof_id+columns[i]%total_num_dof;
//  (dynamic_cast<matrix_solve*>(mat))->set_row(row_g, *numVals, &columns_g[0], values);
  int err = mat->epetra_mat->SumIntoGlobalValues(row_g, *numVals, values, &columns_g[0]);
  if (err) 
    err =mat->epetra_mat->InsertGlobalValues(row_g, *numVals, values, &columns_g[0]);
  return M3DC1_SUCCESS;
}

int m3dc1_epetra_print(int* matrix_id)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat)
    return M3DC1_FAILURE;

  // assemble matrix
  Epetra_Export exporter(/*target*/*(mat->_overlap_map),/*source*/*(mat->_owned_map));
  Epetra_CrsMatrix A(Copy, *(mat->_owned_map), mat->nge);
  A.Export(*(mat->epetra_mat),exporter,Add);
  A.FillComplete();
  A.OptimizeStorage();
  A.MakeDataContiguous();
  A.Print(cout);
  return M3DC1_SUCCESS;
}

int m3dc1_epetra_write(int* matrix_id, const char* filename, int* skip_zero, int* start_index)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat)
    return M3DC1_FAILURE;
  if (!filename)
    return m3dc1_epetra_print(matrix_id);

  char matrix_filename[256];
  sprintf(matrix_filename,"%s-%d",filename, PCU_Comm_Self());
  if (*skip_zero==0)
    write_matrix(mat->epetra_mat, matrix_filename,false,*start_index);
  else
    write_matrix(mat->epetra_mat, matrix_filename,true,*start_index);

  // assemble matrix
  Epetra_Export exporter(*(mat->_overlap_map),*(mat->_owned_map));
  Epetra_CrsMatrix A(Copy, *(mat->_owned_map), mat->nge);
  A.Export(*(mat->epetra_mat),exporter,Add);
  A.FillComplete();
  A.OptimizeStorage();
  A.MakeDataContiguous();

  sprintf(matrix_filename,"assembled-%s-%d",filename, PCU_Comm_Self());

  if (*skip_zero==0)
    write_matrix(&A, matrix_filename,false,*start_index);
  else
    write_matrix(&A, matrix_filename,true,*start_index);

  return M3DC1_SUCCESS;
}

void copyEpetraVec2Field(Epetra_MultiVector* x, apf::Field* f)
{
  int start_global_dofid, num_dof = countComponents(f);
  std::vector<double> dof_data(num_dof);
  apf::Mesh2* m = m3dc1_mesh::instance()->mesh;
  apf::MeshEntity* e;

  int index=0;
  apf::MeshIterator* it = m->begin(0);
  while ((e = m->iterate(it)))
  {
    if (get_ent_ownpartid(m, e)!=PCU_Comm_Self()) continue;
    for(int i=0; i<num_dof; ++i)
      dof_data.at(i)=(*x)[0][index++];
    setComponents(f, e, 0, &(dof_data[0]));
  }
  m->end(it);
  assert(index == num_dof*m3dc1_mesh::instance()->num_own_ent[0]);
  synchronize_field(f);
}

bool isNotAlnum(char c) {
    return isalnum(c) == 0;
}

// For filtering spaces and control characters in Trilinos
// options
bool invalidChar (char c) 
{  
  return !((c > 65 && c < 90) ||
	   (c > 97 && c < 122) ||
	   (c == '_'));
} 

int m3dc1_solver_aztec(int* matrix_id, FieldID* x_fieldid, FieldID*
		       b_fieldid, int* num_iter, double* tolerance,
		       const char* krylov_solver, const char*
		       preconditioner, const char* sub_dom_solver,
		       int* overlap, int* graph_fill, double*
		       ilu_drop_tol, double* ilu_fill, double*
		       ilu_omega, int* poly_ord)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat || mat->matrix_type!=M3DC1_SOLVE)
  {
    std::cout <<"[M3D-C1 ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
    return M3DC1_FAILURE;
  }
  else
    if (!PCU_Comm_Self())
	std::cout <<"[M3D-C1 INFO] "<<__func__<<": matrix "<<*
	matrix_id<<", field "<<* x_fieldid<<" (tol "<<*tolerance<<")\n";

  // assemble matrix
  Epetra_Export exporter(/*target*/*(mat->_overlap_map),
			 /*source*/*(mat->_owned_map));
  Epetra_CrsMatrix A(Copy, *(mat->_owned_map), mat->nge);
  A.Export(*(mat->epetra_mat),exporter,Add);
  A.FillComplete();
  A.OptimizeStorage();
  A.MakeDataContiguous();

  // copy field to vec  
  apf::Field* b_field = (*(m3dc1_mesh::instance()->field_container))[*b_fieldid]->get_field();
  synchronize_field(b_field);
  double* b_field_data = getArrayData(b_field);

  Epetra_MultiVector b_field_vec(*(mat->_overlap_map), 1);
  for (int i=0; i<b_field_vec.MyLength(); ++i)
    b_field_vec[0][i] = b_field_data[i];

  Epetra_MultiVector b(*(mat->_owned_map), 1);
  b.Export(b_field_vec,exporter,Insert);

  // vector for solution
  Epetra_MultiVector x(*(mat->_owned_map), 1);

  Epetra_LinearProblem problem(&A,&x,&b);
  AztecOO solver(problem);
  solver.SetAztecOption(AZ_output,1);

  // Setup solver from input/default

  // Convert const char* to string for comparison
  std::string krylov_solver_s = krylov_solver;
  std::string preconditioner_s = preconditioner;
  std::string sub_dom_solver_s = sub_dom_solver;

  krylov_solver_s.erase(std::remove_if(krylov_solver_s.begin(),
				       krylov_solver_s.end(),
				       invalidChar),
			krylov_solver_s.end());
  preconditioner_s.erase(std::remove_if(preconditioner_s.begin(),
					preconditioner_s.end(),
					invalidChar),
			 preconditioner_s.end());
  sub_dom_solver_s.erase(std::remove_if(sub_dom_solver_s.begin(),
					sub_dom_solver_s.end(),
					invalidChar),
			 sub_dom_solver_s.end());
  
  if (krylov_solver_s == "cg")
    solver.SetAztecOption(AZ_solver, AZ_cg);

  if (krylov_solver_s == "cg_condnum")
    solver.SetAztecOption(AZ_solver, AZ_cg_condnum);

  if (krylov_solver_s == "gmres")
    solver.SetAztecOption(AZ_solver, AZ_gmres);

  if (krylov_solver_s == "gmres_condnum")
    solver.SetAztecOption(AZ_solver, AZ_gmres_condnum);

  if (krylov_solver_s == "cgs")
    solver.SetAztecOption(AZ_solver, AZ_cgs);

  if (krylov_solver_s == "tfqmr")
    solver.SetAztecOption(AZ_solver, AZ_tfqmr);

  // Setup preconditioner from input/default
  if (preconditioner_s == "none")
    solver.SetAztecOption(AZ_precond, AZ_none);

  if (preconditioner_s == "Jacobi")
    solver.SetAztecOption(AZ_precond, AZ_Jacobi);

  if (preconditioner_s == "Neumann")
    solver.SetAztecOption(AZ_precond, AZ_Neumann);

  if (preconditioner_s == "ls")
    solver.SetAztecOption(AZ_precond, AZ_ls);

  if (preconditioner_s == "sym_GS")
    solver.SetAztecOption(AZ_precond, AZ_sym_GS);

  if (preconditioner_s == "dom_decomp")
    solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
  
  // Setup subdomain solver from input/default
  if (preconditioner_s == "dom_decomp")
    {
      if (sub_dom_solver_s == "ilu")
	solver.SetAztecOption(AZ_subdomain_solve, AZ_ilu);

      if (sub_dom_solver_s == "lu")
	solver.SetAztecOption(AZ_subdomain_solve, AZ_lu);

      if (sub_dom_solver_s == "ilut")
	solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);

      if (sub_dom_solver_s == "rilu")
	solver.SetAztecOption(AZ_subdomain_solve, AZ_rilu);

      if (sub_dom_solver_s == "bilu")
	solver.SetAztecOption(AZ_subdomain_solve, AZ_bilu);

      if (sub_dom_solver_s == "icc")
	solver.SetAztecOption(AZ_subdomain_solve, AZ_icc);
      
      // Set Aztec options from input for dom_decomp
      solver.SetAztecOption(AZ_overlap, *overlap);
      solver.SetAztecOption(AZ_graph_fill, *graph_fill);

      // Setup Aztec parameters from input/default
      solver.SetAztecParam(AZ_tol, *tolerance);
      solver.SetAztecParam(AZ_drop, *ilu_drop_tol);
      solver.SetAztecParam(AZ_ilut_fill, *ilu_fill);
      if (sub_dom_solver_s == "rilu")
	solver.SetAztecParam(AZ_omega, *ilu_omega);
    }
  
  // Setup alternate preconditioner options from input/default
  if (preconditioner_s == "Jacobi" ||
      preconditioner_s == "Neumann" ||
      preconditioner_s == "ls" ||
      preconditioner_s == "sym_GS")
    solver.SetAztecOption(AZ_poly_ord, *poly_ord);

  // Now perform the solve
  solver.Iterate(*num_iter,*tolerance);
  mat->num_solver_iter = solver.NumIters();
  
  apf::Field* x_field = (*(m3dc1_mesh::instance()->field_container))[*x_fieldid]->get_field();
  copyEpetraVec2Field(&x, x_field);
  return M3DC1_SUCCESS;
}

// solve Ax=b
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_CommandLineProcessor.hpp>

int m3dc1_solver_amesos(int* matrix_id, FieldID* x_fieldid, FieldID* b_fieldid, const char* solver_name)
{
  if (!PCU_Comm_Self()) std::cout <<"[M3D-C1 ERROR] "<<__func__<<" not supported: check if amesos2 is available\n";
  return M3DC1_FAILURE;

  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat || mat->matrix_type!=M3DC1_SOLVE)
  {
    std::cout <<"[M3D-C1 ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
    return M3DC1_FAILURE;
  }

  Epetra_Export exporter(*(mat->_overlap_map),*(mat->_owned_map));
  Epetra_CrsMatrix A(Copy, *(mat->_owned_map), mat->nge);
  
  apf::Field* b_field = (*(m3dc1_mesh::instance()->field_container))[*b_fieldid]->get_field();
  double* b_field_data = getArrayData(b_field);

  Epetra_MultiVector b_field_vec(*(mat->_overlap_map), 1);
  // copy field to vec
  for (int i=0; i<b_field_vec.MyLength(); ++i)
    b_field_vec[0][i] = b_field_data[i];

  Epetra_MultiVector b(*(mat->_owned_map), 1 );

  A.Export(*(mat->epetra_mat),exporter,Add);
  b.Export(b_field_vec,exporter,Insert);

  A.FillComplete();
  A.OptimizeStorage();
  A.MakeDataContiguous();

  return M3DC1_SUCCESS;
}

int m3dc1_solver_getnumiter(int* matrix_id, int * num_iter)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat)
    return M3DC1_FAILURE;
  *num_iter = mat->num_solver_iter;
  return M3DC1_SUCCESS;
}

// local matrix multiplication
// do accumulate(out_field) for global result
int m3dc1_epetra_multiply(int* matrix_id, FieldID* in_fieldid, FieldID* out_fieldid)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat || mat->matrix_type!=M3DC1_MULTIPLY)
  {
    std::cout <<"[M3D-C1 ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
    return M3DC1_FAILURE;
  }

  if (!mat->epetra_mat->Filled())
    mat->epetra_mat->FillComplete();
  assert(mat->epetra_mat->Filled());

  apf::Field* in_field = (*(m3dc1_mesh::instance()->field_container))[*in_fieldid]->get_field();
  double* in_field_data =getArrayData(in_field);

  Epetra_MultiVector x(mat->epetra_mat->RowMap(), 1);
  // copy field to vec
  for (int i=0; i<x.MyLength(); ++i)
    x[0][i] = in_field_data[i];
  Epetra_MultiVector b(mat->epetra_mat->RowMap(), 1);
  EPETRA_CHK_ERR(mat->epetra_mat->Multiply(false, x, b));
  apf::Field* out_field = (*(m3dc1_mesh::instance()->field_container))[*out_fieldid]->get_field();
  double* out_field_data =getArrayData(out_field);
  b.ExtractCopy(out_field_data, b.MyLength());
  accumulate(out_field);
  return M3DC1_SUCCESS;
}

int m3dc1_epetra_freeze(int* matrix_id)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat)
    return M3DC1_FAILURE;
  mat->epetra_mat->FillComplete();
  assert(mat->epetra_mat->Filled());
  return M3DC1_SUCCESS;
}
#endif
