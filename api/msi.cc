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

void set_adj_node_tag(pMeshTag num_global_adj_node_tag, pMeshTag num_own_adj_node_tag);

//*******************************************************
void msi_ment_getFieldID (pMeshEnt e, pField f, int inode,
     int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif
  // FIXME: replace getMdsIndex with numbering
  int ent_id = getMdsIndex(pumi::instance()->mesh, e);
  *start_dof_id = ent_id*num_dof;
  *end_dof_id_plus_one = *start_dof_id +num_dof;
}

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

//*******************************************************
void msi_ment_getGlobalFieldID (pMeshEnt e, pField f, int inode,
     int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif
  // FIXME: replace getMdsIndex with numbering
  int ent_id = pumi_ment_getGlobalID(e);
  *start_dof_id = ent_id*num_dof;
  *end_dof_id_plus_one = *start_dof_id +num_dof;
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
 
//*******************************************************
void msi_field_getinfo(pField f, 
                        char* /* out*/ field_name, int* num_values, 
                        int* total_num_dof)
//*******************************************************
{
  strcpy(field_name, getName(f));
  *num_values = 1;
  *total_num_dof = countComponents(f);
#ifdef PETSC_USE_COMPLEX
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

  PetscMemorySetGetMaximumUsage();
  msi_solver::instance()->num_global_adj_node_tag = pumi::instance()->mesh->createIntTag("msi_num_global_adj_node", 1);
  msi_solver::instance()->num_own_adj_node_tag = pumi::instance()->mesh->createIntTag("msi_num_own_adj_node", 1);
  set_adj_node_tag(msi_solver::instance()->num_global_adj_node_tag, msi_solver::instance()->num_own_adj_node_tag);

}

void msi_finalize(pMesh m)
{  
  apf::removeTagFromDimension(m, msi_solver::instance()->num_global_adj_node_tag, 0);
  m->destroyTag(msi_solver::instance()->num_global_adj_node_tag);
  apf::removeTagFromDimension(m, msi_solver::instance()->num_own_adj_node_tag, 0);
  m->destroyTag(msi_solver::instance()->num_own_adj_node_tag);

  while(m->countFields())
  {
    apf::Field* f = m->getField(0);
    if(!PCU_Comm_Self()) std::cout<<"[MSI INFO] "<<__func__<<": field "<<getName(f)<<" deleted\n";
    destroyField(f);
  }

  pumi_mesh_deleteGlobalID(m);  // delete global id
}



//*******************************************************
void msi_field_assign(pField f, double* fac)
//*******************************************************
{
  int scalar_type=0, dofPerEnt = countComponents(f);
#ifdef PETSC_USE_COMPLEX
  dofPerEnt /= 2;
  scalar_type=1;
#endif

  int vertex_type=0, num_vtx=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  std::vector<double> dofs(dofPerEnt*(1+scalar_type), fac[0]);
  if(scalar_type)
    for(int i=0; i<dofPerEnt; i++)
      dofs.at(2*i+1)=fac[1];
  for(int inode=0; inode<num_vtx; inode++)
    msi_ent_setdofdata (&vertex_type, &inode, f, &dofPerEnt, &dofs[0]);
}


pField msi_field_create (pMesh m, const char* /* in */ field_name, int /*in*/ num_values, int /*in*/ num_dofs_per_value)
{
  int scalar_type=0;
#ifdef PETSC_USE_COMPLEX
  scalar_type=1;
#endif
  int components = num_values*(scalar_type+1)*num_dofs_per_value;
  apf::Field* f = createPackedField(m, field_name, components);
  apf::freeze(f); // switch dof data from tag to array
  double val[2]={0,0};
  msi_field_assign(f, val);
  return f;
}

#ifdef MSI_PETSC
/** matrix and solver functions */
//*******************************************************
msi_matrix* msi_matrix_create(int matrix_type, pField f)
//*******************************************************
{
  if (matrix_type==MSI_MULTIPLY) // matrix for multiplication
  {
    matrix_mult* new_mat = new matrix_mult(f);
    return (msi_matrix*)new_mat;
  }
  else 
  {
    matrix_solve* new_mat= new matrix_solve(f);
    return (msi_matrix*)new_mat;
  }

#ifdef DEBUG
  if (!PCU_Comm_Self())
    std::cout<<"[MSI INFO] "<<__func__<<": type "<<matrix_type<<", field "<<getName(f)<<"\n";
#endif 
}

//*******************************************************
void msi_matrix_assemble(pMatrix mat) 
//*******************************************************
{
  mat->assemble();
}

//*******************************************************
void msi_matrix_delete(pMatrix mat)
//*******************************************************
{  
  delete mat;
}

//*******************************************************
void msi_matrix_insert(pMatrix mat, int row, 
         int col, int scalar_type, double* val)
//*******************************************************
{  
  assert(mat->get_status()!=MSI_FIXED);

#ifdef DEBUG
  int num_values, total_num_dof;
  char field_name[256];
  msi_field_getinfo(mat->get_field(), field_name, &num_values, &total_num_dof);

  int ent_id = row/total_num_dof;
  apf::MeshEntity* e =apf::getMdsEntity(pumi::instance()->mesh, 0, ent_id);
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));
#endif

  if (scalar_type) // complex
    mat->set_value(row, col, INSERT_VALUES, val[0], val[1]);
  else
    mat->set_value(row, col, INSERT_VALUES, *val, 0);
}

//*******************************************************
void msi_matrix_add (pMatrix mat, int row, int col, 
                    int scalar_type, double* val) 
//*******************************************************
{  
  assert(mat->get_status()!=MSI_FIXED);

#ifdef DEBUG
  int num_values, total_num_dof;
  char field_name[256];
  msi_field_getinfo(mat->get_field(), field_name, &num_values, &total_num_dof);

  int ent_id = row/total_num_dof;
  apf::MeshEntity* e =apf::getMdsEntity(pumi::instance()->mesh, 0, ent_id);
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));
#endif

  if (scalar_type) // complex
    mat->set_value(row, col, ADD_VALUES, val[0], val[1]);
  else
    mat->set_value(row, col, ADD_VALUES, *val, 0);
}

//*******************************************************
void msi_matrix_setBC(pMatrix mat, int row)
//*******************************************************
{  
  assert(mat->get_type()==MSI_SOLVE);

  int num_values, total_num_dof;
  char field_name[256];
  msi_field_getinfo(mat->get_field(), field_name, &num_values, &total_num_dof);
  int inode = row/total_num_dof;
  int ent_dim=0, start_global_dof_id, end_global_dof_id_plus_one;
  msi_ent_getglobaldofid (&ent_dim, &inode, mat->get_field(), &start_global_dof_id, &end_global_dof_id_plus_one);

#ifdef DEBUG
  apf::MeshEntity* e =apf::getMdsEntity(pumi::instance()->mesh, 0, inode);
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));

  int start_dof_id, end_dof_id_plus_one;
  msi_ent_getlocaldofid (&ent_dim, &inode, mat->get_field(), &start_dof_id, &end_dof_id_plus_one);
  assert(row>=start_dof_id&&row<end_dof_id_plus_one);
#endif
  int row_g = start_global_dof_id+row%total_num_dof;
  (dynamic_cast<matrix_solve*>(mat))->set_bc(row_g);
}

//*******************************************************
void msi_matrix_setLaplaceBC(pMatrix mat, int row,
         int numVals, int* columns, double* values)
//*******************************************************
{
  assert(mat->get_type()==MSI_SOLVE);

  std::vector <int> columns_g(numVals);
  int num_values, total_num_dof;
  char field_name[256];
  msi_field_getinfo(mat->get_field(), field_name, &num_values, &total_num_dof);
  int inode = row/total_num_dof;
  int ent_dim=0, start_global_dof_id, end_global_dof_id_plus_one;
  msi_ent_getglobaldofid (&ent_dim, &inode, mat->get_field(), &start_global_dof_id, &end_global_dof_id_plus_one);

#ifdef DEBUG
  apf::MeshEntity* e =apf::getMdsEntity(pumi::instance()->mesh, 0, inode);
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));
  int start_dof_id, end_dof_id_plus_one;
  msi_ent_getlocaldofid (&ent_dim, &inode, mat->get_field(), &start_dof_id, &end_dof_id_plus_one);
  assert(row>=start_dof_id&&row<end_dof_id_plus_one);
  for (int i=0; i<numVals; i++)
    assert(columns[i]>=start_dof_id&&columns[i]<end_dof_id_plus_one);
#endif

  int row_g = start_global_dof_id+row%total_num_dof;
  for(int i=0; i<numVals; i++)
  {
    columns_g.at(i) = start_global_dof_id+columns[i]%total_num_dof;
  }
  (dynamic_cast<matrix_solve*>(mat))->set_row(row_g, numVals, &columns_g[0], values);
}

void msi_matrix_solve(pMatrix mat, pField rhs, pField sol) 
{  
  assert(mat->get_type()==MSI_SOLVE);

  if (!PCU_Comm_Self())
     std::cout <<"[M3D-C1 INFO] "<<__func__<<": RHS "<<getName(rhs)<<", solution: "<<getName(sol)<<"\n";

  (dynamic_cast<matrix_solve*>(mat))->solve(rhs, sol);
}

//*******************************************************
void msi_matrix_multiply(pMatrix mat, pField inputvec, pField outputvec) 
//*******************************************************
{  
  assert(mat->get_type()==MSI_MULTIPLY);

  (dynamic_cast<matrix_mult*>(mat))->multiply(inputvec, outputvec);
}

//*******************************************************
void msi_matrix_flush(pMatrix mat)
//*******************************************************
{
  mat->flushAssembly();
}

//*******************************************************
int msi_matrix_getNumIter(pMatrix mat)
//*******************************************************
{ 
  return dynamic_cast<matrix_solve*> (mat)->iterNum;
}

//*******************************************************
void msi_matrix_addBlock(pMatrix mat, int ielm, 
          int rowIdx, int columnIdx, double* values)
//*******************************************************
{
  // need to change later, should get the value from field calls ...
  int dofPerVar = 6;
  char field_name[256];
  int num_values, total_num_dof; 
  msi_field_getinfo(mat->get_field(), field_name, &num_values, &total_num_dof);
  dofPerVar=total_num_dof/num_values;
  int nodes[6];
  int ent_dim=0;
  int ielm_dim = 2;
  int nodes_per_element=sizeof(nodes)/sizeof(int), nodes_per_element_get;

  if (pumi::instance()->mesh->getDimension()==3) ielm_dim =3;

  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, ielm_dim, ielm);
  assert(e);
  if (pumi::instance()->mesh->isGhost(e)) return;
  
  msi_ent_getadj (&ielm_dim, &ielm, &ent_dim, nodes, &nodes_per_element, &nodes_per_element_get);
  nodes_per_element=nodes_per_element_get;
  int start_global_dof_id,end_global_dof_id_plus_one;
  int start_global_dof,end_global_dof_id;
  // need to change later, should get the value from field calls ...
  int scalar_type = 0;
#ifdef PETSC_USE_COMPLEX
  scalar_type=1;
#endif

  int numDofs = total_num_dof;
  int numVar = numDofs/dofPerVar;
  assert(rowIdx<numVar && columnIdx<numVar);
  int rows[1024], columns[1024];
  assert(sizeof(rows)/sizeof(int)>=dofPerVar*nodes_per_element);
  if(mat->get_type()==0)
  {
    int localFlag=0;
    matrix_mult* mmat = dynamic_cast<matrix_mult*> (mat);
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      if (mmat->is_mat_local()) 
        msi_ent_getlocaldofid (&ent_dim, nodes+inode, mat->get_field(), &start_global_dof_id, &end_global_dof_id_plus_one);
      else 
        msi_ent_getglobaldofid (&ent_dim, nodes+inode, mat->get_field(), &start_global_dof_id, &end_global_dof_id_plus_one);
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+rowIdx*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+columnIdx*dofPerVar+i;
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
      msi_ent_getglobaldofid (&ent_dim, nodes+inode, mat->get_field(), &start_global_dof_id, &end_global_dof_id_plus_one);
      rows_bloc[inode]=nodes[inode]*numVar+rowIdx;
      columns_bloc[inode]=nodes[inode]*numVar+columnIdx;
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+rowIdx*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+columnIdx*dofPerVar+i;
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
void msi_matrix_write(pMatrix mat, const char* filename, int start_index)
//*******************************************************
{
  if (!filename) 
    return msi_matrix_print(mat);

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
      fprintf(fp, "%d\t%d\t%E\n", row+start_index, cols[index]+start_index,vals[index]);
      ++index;
    }
  }
  fclose(fp);  
  assert(index == vals.size());
}


//*******************************************************
void msi_matrix_print(pMatrix mat)
//*******************************************************
{

  int row, col, csize, sum_csize=0, index=0;

  vector<int> rows;
  vector<int> n_cols;
  vector<int> cols;
  vector<double> vals;

  mat->get_values(rows, n_cols, cols, vals);
  for (int i=0; i<rows.size(); ++i)
    sum_csize += n_cols[i];
  assert(vals.size()==sum_csize);

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

int m3dc1_epetra_create(int* matrix_id, int* matrix_type, pField f)
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
  m3dc1_ls::instance()->add_matrix(*matrix_id, new m3dc1_epetra(*matrix_id, *matrix_type, *field_id));
  return M3DC1_SUCCESS;
}

int m3dc1_epetra_delete(int* matrix_id)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  assert(mat);

  typedef std::map<int, m3dc1_epetra*> matrix_container_map;
  m3dc1_ls::instance()->matrix_container->erase(matrix_container_map::key_type(*matrix_id));
  mat->destroy();
  delete mat;
  return M3DC1_SUCCESS;
}

int m3dc1_epetra_insert(int* matrix_id, int* row, int* col, double* val)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  assert(mat);

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

  assert(mat);

  int field = mat->get_field_id();
  // need to change later, should get the value from field calls ...
  int dofPerVar = 6;
  char field_name[256];
  int num_values, total_num_dof; 
  m3dc1_field_getinfo(&field, field_name, &num_values, &total_num_dof);
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
  int scalar_type = 0;
#ifdef MSI_COMPLEX
  scalar_type=1;
#endif

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
  int num_values, total_num_dof;
  char field_name[256];
  m3dc1_field_getinfo(&field, field_name, &num_values, &total_num_dof);
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
  int num_values, total_num_dof;
  char field_name[256];
  m3dc1_field_getinfo(&field, field_name, &num_values, &total_num_dof);
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

  int err = mat->epetra_mat->SumIntoGlobalValues(row_g, *numVals, values, &columns_g[0]);
  if (err) 
    err =mat->epetra_mat->InsertGlobalValues(row_g, *numVals, values, &columns_g[0]);
  return M3DC1_SUCCESS;
}

int m3dc1_epetra_print(int* matrix_id)
{
  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  assert(mat);

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
  assert(mat);

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

int m3dc1_solver_aztec(int* matrix_id, pField x_field, pField b_field, 
                       int* num_iter, double* tolerance,
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
  assert(mat);

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
  assert(mat);
  
  mat->epetra_mat->FillComplete();
  assert(mat->epetra_mat->Filled());
  return M3DC1_SUCCESS;
}
#endif

#ifdef MSI_PETSC
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
#endif
