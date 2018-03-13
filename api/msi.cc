/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#include "msi.h"
#include "msi_solver.h"
#include "msi_petsc.h"
#include <iostream>
#include "PCU.h"
#include "apfMDS.h"
#include <vector>
#include <assert.h>
#include "apfShape.h"
#include "apfNumbering.h"
using std::vector;

void set_adj_node_tag(pMesh m, pOwnership, pMeshTag num_global_adj_node_tag, pMeshTag num_own_adj_node_tag);

// returns sequential local numbering of entity's ith node
// local numbering is based on mesh shape 
int msi_node_getID (pMeshEnt e, int n)
{
  assert(apf::isNumbered(msi_solver::instance()->local_n,e,n,0));
  return pumi_node_getNumber(msi_solver::instance()->local_n, e, n);
}

// returns global numbering of entity's ith node 
// global numbering is based on ownership set in msi_start
int msi_node_getGlobalID (pMeshEnt e, int n)
{
  assert(apf::isNumbered(msi_solver::instance()->global_n,e,n,0));
  return pumi_node_getNumber(msi_solver::instance()->global_n, e, n);
}

#define FIELDVALUELIMIT 1e100
bool value_is_nan(double val)
{
  return val!=val ||fabs(val) >FIELDVALUELIMIT;
}

void msi_node_setField (pField f, pMeshEnt e, int n, int num_dof, double* dof_data)
{
#ifdef DEBUG
  int scalar_type=0;
#ifdef PETSC_USE_COMPLEX
  scalar_type=1;
#endif
  assert(countComponents(f)==num_dof*(1+scalar_type));
  for(int i=0; i<num_dof*(1+scalar_type); ++i)
    assert(!value_is_nan(dof_data[i]));
#endif
  setComponents(f, e, n, dof_data);
}

int msi_node_getField(pField f, pMeshEnt e, int n, double* dof_data)
{
  getComponents(f, e, n, dof_data);
  int num_dof=apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof /=2;
#endif

#ifdef DEBUG
  int scalar_type=0;
#ifdef PETSC_USE_COMPLEX
  scalar_type=1;
#endif
  for(int i=0; i<num_dof*(1+scalar_type); i++)
    assert(!value_is_nan(dof_data[i]));
  int start_dof_id,end_dof_id_plus_one;
  msi_node_getFieldID(f, e, n, &start_dof_id, &end_dof_id_plus_one);
  double* data;
  msi_field_getdataptr(f, &data);
  int start=start_dof_id*(1+scalar_type);
  for(int i=0; i<num_dof; i++)
    assert(data[start++]==dof_data[i]);
#endif
  return num_dof;
}

//*******************************************************
void msi_node_getFieldID (pField f, pMeshEnt e, int n,
     int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif
  int ent_id = msi_node_getID(e, n);
  *start_dof_id = ent_id*num_dof;
  *end_dof_id_plus_one = *start_dof_id +num_dof;
}

//*******************************************************
void msi_node_getGlobalFieldID (pField f, pMeshEnt e, int n,
     int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof/=2;
#endif
  int ent_id = msi_node_getGlobalID(e, n);
  *start_dof_id = ent_id*num_dof;
  *end_dof_id_plus_one = *start_dof_id +num_dof;
}
#include <unistd.h>
//********************************************************
void msi_start(pMesh m, pOwnership o, pShape s, MPI_Comm cm)
//********************************************************
{  
#if 0 // turn on to debug with gdb
  int i, processid = getpid();
  if (!PCU_Comm_Self())
  {
    std::cout<<"Proc "<<PCU_Comm_Self()<<">> pid "<<processid<<" Enter any digit...\n";
    std::cin>>i;
  }
  else
    std::cout<<"Proc "<<PCU_Comm_Self()<<">> pid "<<processid<<" Waiting...\n";
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (o) 
  {
    msi_solver::instance()->ownership=o;
    if (!pumi_rank()) std::cout<<"[msi] ("<<pumi_rank()<<") "<<__func__<<": user-defined ownership is in use\n";
    pumi_ownership_verify(m, o);
  }
  else
  {
    msi_solver::instance()->ownership=new apf::NormalSharing(m);
    if (!pumi_rank()) std::cout<<"[msi] ("<<pumi_rank()<<") "<<__func__<<": the default mesh ownership is in use\n";
  }

  pumi_mesh_setCount(m, o);

  if (s) 
    pumi_mesh_setShape(m, s);
  else
    s = pumi_mesh_getShape(m);

  PetscMemorySetGetMaximumUsage();
  msi_solver::instance()->num_global_adj_node_tag = m->createIntTag("msi_num_global_adj_node", 1);
  msi_solver::instance()->num_own_adj_node_tag = m->createIntTag("msi_num_own_adj_node", 1);
  set_adj_node_tag(m, o, msi_solver::instance()->num_global_adj_node_tag, 
                   msi_solver::instance()->num_own_adj_node_tag);

  // set local numbering
  const char* name = s->getName();
  pNumbering ln = m->findNumbering(name);
  if (!ln) 
    ln = apf::numberOverlapNodes(m,name,s); 
  msi_solver::instance()->local_n = ln;

  // generate global ID's per ownership
  if(cm == MPI_Comm_NULL)
    msi_solver::instance()->global_n = pumi_numbering_createGlobal(m, "pumi_global", NULL, o);
  else
    msi_solver::instance()->global_n = msi_numbering_createGlobal_degenerated(m, "pumi_global", NULL, o, cm);

  msi_solver::instance()->vertices = new pMeshEnt[m->count(0)];

  pMeshEnt e;
  pMeshIter it =m->begin(0);  
  while ((e = m->iterate(it)))
  {
#ifdef DEBUG
    assert(apf::isNumbered(msi_solver::instance()->local_n,e,0,0));
    assert(apf::isNumbered(msi_solver::instance()->global_n,e,0,0));
#endif
    msi_solver::instance()->vertices[msi_node_getID(e, 0)] = e;
  }
  m->end(it);
}

pNumbering msi_numbering_createGlobal_degenerated(pMesh m, const char* name, pShape s, pOwnership o, MPI_Comm cm)
{
  pNumbering n = m->findNumbering(name);
  if (n)
  {
    if (!pumi_rank())
      std::cout<<"[PUMI INFO] "<<__func__<<" failed: numbering \""<<name<<"\" already exists\n";
    return n;
  }

  if (!s) s= m->getShape();
  n = numberOwnedNodes(m, name, s, o);

  PCU_Switch_Comm(cm);
  apf::globalize(n);
  PCU_Switch_Comm(MPI_COMM_WORLD);

  //apf::synchronizeFieldData<int>(n->getData(), o, false); //synchronize(n, o);
  return n;
}

void msi_finalize(pMesh m)
{  
  apf::removeTagFromDimension(m, msi_solver::instance()->num_global_adj_node_tag, 0);
  m->destroyTag(msi_solver::instance()->num_global_adj_node_tag);
  apf::removeTagFromDimension(m, msi_solver::instance()->num_own_adj_node_tag, 0);
  m->destroyTag(msi_solver::instance()->num_own_adj_node_tag);

  pumi_numbering_delete(msi_solver::instance()->local_n);
  pumi_numbering_delete(msi_solver::instance()->global_n);  
}

pOwnership msi_getOwnership()
{
  return msi_solver::instance()->ownership;
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
  std::vector<double> dofs(dofPerEnt*(1+scalar_type), fac[0]);
  if(scalar_type)
    for(int i=0; i<dofPerEnt; i++)
      dofs.at(2*i+1)=fac[1];

  pMeshEnt e;
  pMeshIter it = pumi::instance()->mesh->begin(0);  
  while ((e = pumi::instance()->mesh->iterate(it)))
    msi_node_setField (f, e, 0, dofPerEnt, &dofs[0]);
  pumi::instance()->mesh->end(it);
}

//*******************************************************
pField msi_field_create (pMesh m, const char* /* in */ field_name, int /*in*/ num_values, 
int /*in*/ num_dofs_per_value, pShape shape)
//*******************************************************
{
  int scalar_type=0;
#ifdef PETSC_USE_COMPLEX
  scalar_type=1;
#endif
  int components = num_values*(scalar_type+1)*num_dofs_per_value;
  apf::Field* f = createPackedField(m, field_name, components, shape);
  msi_solver::instance()->add_field(f, num_values);
  apf::freeze(f); // switch dof data from tag to array
  double val[2]={0,0};
  msi_field_assign(f, val);
  return f;
}

//*******************************************************
int msi_field_getNumVal(pField f)
//*******************************************************
{
  return (*msi_solver::instance()->field_container)[f];
}

//*******************************************************
int msi_field_getSize(pField f)
//*******************************************************
{
#ifdef PETSC_USE_COMPLEX
  return countComponents(f)/2;
#else
  return countComponents(f);
#endif
}

#ifdef MSI_PETSC
void msi_matrix_setComm(MPI_Comm cm)
{
  PETSC_COMM_WORLD = cm;
}

/** matrix and solver functions */
//*******************************************************
msi_matrix* msi_matrix_create(int matrix_type, pField f)
//*******************************************************
{
#ifdef DEBUG
  if (!PCU_Comm_Self())
    std::cout<<"[msi] "<<__func__<<": type "<<matrix_type<<", field "<<getName(f)<<"\n";
#endif
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
}

pField msi_matrix_getField(pMatrix mat)
{
  return mat->get_field();
}


//*******************************************************
void msi_matrix_assemble(pMatrix mat) 
//*******************************************************
{
#ifdef DEBUG
  if (!PCU_Comm_Self()) std::cout<<"[msi] "<<__func__<<"\n";
#endif
  mat->assemble();
}

//*******************************************************
void msi_matrix_delete(pMatrix mat)
//*******************************************************
{  
#ifdef DEBUG
  if (!PCU_Comm_Self()) std::cout<<"[msi] "<<__func__<<"\n";
#endif
  delete mat;
}

//*******************************************************
void msi_matrix_insert(pMatrix mat, int row, 
         int col, int scalar_type, double* val)
//*******************************************************
{  
  assert(mat->get_status()!=MSI_FIXED);

#ifdef DEBUG
  int num_values = msi_field_getNumVal(mat->get_field());
  int total_num_dof = msi_field_getSize(mat->get_field());

  int ent_id = row/total_num_dof;
  apf::MeshEntity* e =msi_solver::instance()->vertices[ent_id];
  assert(e && !pumi::instance()->mesh->isGhost(e));
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
  int num_values = msi_field_getNumVal(mat->get_field());
  int total_num_dof = msi_field_getSize(mat->get_field());

  int ent_id = row/total_num_dof;
  apf::MeshEntity* e =msi_solver::instance()->vertices[ent_id];
  assert(e&&!pumi::instance()->mesh->isGhost(e));
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

  int num_values = msi_field_getNumVal(mat->get_field());
  int total_num_dof = msi_field_getSize(mat->get_field());

  int inode = row/total_num_dof;
  pMeshEnt e = msi_solver::instance()->vertices[inode];
  int start_global_dof_id, end_global_dof_id_plus_one;
  msi_node_getGlobalFieldID(mat->get_field(), e, 0, &start_global_dof_id, &end_global_dof_id_plus_one);

#ifdef DEBUG
  assert(!pumi::instance()->mesh->isGhost(e));

  int start_dof_id, end_dof_id_plus_one;
  msi_node_getFieldID(mat->get_field(), e, 0, &start_dof_id, &end_dof_id_plus_one);
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

  int num_values = msi_field_getNumVal(mat->get_field());
  int total_num_dof = msi_field_getSize(mat->get_field());

  int inode = row/total_num_dof;
  pMeshEnt e = msi_solver::instance()->vertices[inode];

  int start_global_dof_id, end_global_dof_id_plus_one;
  msi_node_getGlobalFieldID(mat->get_field(), e, 0, 
                            &start_global_dof_id, &end_global_dof_id_plus_one);

#ifdef DEBUG
  assert(!pumi::instance()->mesh->isGhost(e));
  int start_dof_id, end_dof_id_plus_one;
  msi_node_getFieldID(mat->get_field(), e, 0, &start_dof_id, &end_dof_id_plus_one);
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

//*******************************************************
void msi_matrix_multiply(pMatrix mat, pField inputvec, pField outputvec) 
//*******************************************************
{  
#ifdef DEBUG
  if (!PCU_Comm_Self()) std::cout<<"[msi] "<<__func__<<": input \""<<getName(inputvec)<<"\", output \""<<getName(outputvec)<<"\"\n";
#endif
  assert(mat->get_type()==MSI_MULTIPLY);

  (dynamic_cast<matrix_mult*>(mat))->multiply(inputvec, outputvec);
}

//*******************************************************
void msi_matrix_solve(pMatrix mat, pField rhs, pField sol) 
//*******************************************************
{  
  assert(mat->get_type()==MSI_SOLVE);
#ifdef DEBUG
  if (!PCU_Comm_Self())
     std::cout <<"[msi] "<<__func__<<": RHS \""<<getName(rhs)<<"\", sol \""<<getName(sol)<<"\"\n";
#endif
  (dynamic_cast<matrix_solve*>(mat))->solve(rhs, sol);
}

//*******************************************************
int msi_matrix_getNumIter(pMatrix mat)
//*******************************************************
{ 
  return dynamic_cast<matrix_solve*> (mat)->iterNum;
}

//*******************************************************
void msi_matrix_addBlock(pMatrix mat, pMeshEnt e, 
          int rowIdx, int columnIdx, double* values)
//*******************************************************
{
  // need to change later, should get the value from field calls ...
  int num_values = msi_field_getNumVal(mat->get_field());
  int total_num_dof = msi_field_getSize(mat->get_field());

  int dofPerVar=total_num_dof/num_values;

  int ent_dim=0;
  int ielm_dim = pumi::instance()->mesh->getDimension();

  if (pumi::instance()->mesh->isGhost(e)) return;
  std::vector<pMeshEnt> vertices;
  pumi_ment_getAdj(e, 0, vertices);

  int num_node=(int)vertices.size();
  int* nodes=new int[num_node];
  for (int i=0; i<num_node; ++i)
    nodes[i] = msi_node_getID(vertices[i], 0);

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
  assert(sizeof(rows)/sizeof(int)>=dofPerVar*num_node);

  if (mat->get_type()==0) // multiply
  {
    int localFlag=0;
    matrix_mult* mmat = dynamic_cast<matrix_mult*> (mat);
    for(int inode=0; inode<num_node; inode++)
    {
      if (mmat->is_mat_local()) 
        msi_node_getFieldID(mat->get_field(), vertices[inode], 0, &start_global_dof_id, &end_global_dof_id_plus_one);
      else 
        msi_node_getGlobalFieldID(mat->get_field(), vertices[inode], 0, &start_global_dof_id, &end_global_dof_id_plus_one);
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+rowIdx*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+columnIdx*dofPerVar+i;
      }
    }
    mmat->add_values(dofPerVar*num_node, rows ,dofPerVar*num_node, columns, values);
  }
  else // solve
  {
    matrix_solve* smat = dynamic_cast<matrix_solve*> (mat);
    int* nodeOwner = new int[num_node];
    int* columns_bloc = new int[num_node];
    int* rows_bloc = new int[num_node];
    for(int inode=0; inode<num_node; inode++)
    {
      nodeOwner[inode] = msi_solver::instance()->ownership->getOwner(vertices[inode]);
      msi_node_getGlobalFieldID(mat->get_field(), vertices[inode], 0, &start_global_dof_id, &end_global_dof_id_plus_one);
      rows_bloc[inode]=nodes[inode]*numVar+rowIdx;
      columns_bloc[inode]=nodes[inode]*numVar+columnIdx;
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+rowIdx*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+columnIdx*dofPerVar+i;
      }
    }
    int numValuesNode = dofPerVar*dofPerVar*num_node*(1+scalar_type);
    int offset=0;
    for(int inode=0; inode<num_node; inode++)
    {
      if (nodeOwner[inode]!=PCU_Comm_Self())
        smat->add_blockvalues(1, rows_bloc+inode, num_node, columns_bloc, values+offset);
      else 
        smat->add_values(dofPerVar, rows+dofPerVar*inode, dofPerVar*num_node, columns, values+offset);
      offset+=numValuesNode;
    }
    delete [] nodeOwner;
    delete [] columns_bloc;
    delete [] rows_bloc;
  }
  delete [] nodes;
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
      std::cout <<"[msi] "<<__func__<<" failed: matrix with id "<<*matrix_id<<" already created\n";
    return M3DC1_FAILURE; 
  }
  // check field exists
  if (!m3dc1_mesh::instance()->field_container || !m3dc1_mesh::instance()->field_container->count(*field_id))
  {
    if (!PCU_Comm_Self())
      std::cout <<"[msi] "<<__func__<<" failed: field with id "<<*field_id<<" doesn't exist\n";
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
  int num_values = msi_field_getNumVal(mat->get_field());
  int total_num_dof = msi_field_getSize(mat->get_field());

  int dofPerVar=total_num_dof/num_values;
  int nodes[6];
  int ent_dim=0;
  int ielm_dim = 2;
  int num_node=sizeof(nodes)/sizeof(int), num_node_get;

  if (m3dc1_mesh::instance()->mesh->getDimension()==3) ielm_dim =3;
  m3dc1_ent_getadj (&ielm_dim, ielm, &ent_dim, nodes, &num_node, &num_node_get);
  num_node=num_node_get;
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
  int* rows = new int[dofPerVar*num_node];
  int* columns = new int[dofPerVar*num_node];

  if (mat->matrix_type==M3DC1_MULTIPLY)
  {
    for(int inode=0; inode<num_node; inode++)
    {
      m3dc1_ent_getglobaldofid (&ent_dim, nodes+inode, &field, &start_global_dof_id, &end_global_dof_id_plus_one);
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+(*rowVarIdx)*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+(*columnVarIdx)*dofPerVar+i;
      }
    }
    //FIXME: mmat->add_values(dofPerVar*num_node, rows,dofPerVar*num_node, columns, values);
    epetra_add_values(mat->epetra_mat, dofPerVar*num_node, 
                      rows,dofPerVar*num_node, columns, values);     
  }
  else //M3DC1_SOLVE
  {
    int nodeOwner[6];
    int columns_bloc[6], rows_bloc[6];
    for(int inode=0; inode<num_node; inode++)
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
    int numValuesNode = dofPerVar*dofPerVar*num_node*(1+scalar_type);
    int offset=0;
    for(int inode=0; inode<num_node; inode++)
    {
      // FIXME: smat->add_values(dofPerVar, rows+dofPerVar*inode, dofPerVar*num_node, columns, values+offset);
      epetra_add_values(mat->epetra_mat, dofPerVar, rows+dofPerVar*inode, 
                       dofPerVar*num_node, columns, values+offset);
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
    std::cout <<"[msi] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
    return M3DC1_FAILURE;
  }

  int field = mat->get_field_id();
  int num_values = msi_field_getNumVal(mat->get_field());
  int total_num_dof = msi_field_getSize(mat->get_field());

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
    std::cout <<"[msi] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
    return M3DC1_FAILURE;
  }

  std::vector <global_ordinal_type> columns_g(*numVals);
  int field = mat->get_field_id();

  int num_values = msi_field_getNumVal(mat->get_field());
  int total_num_dof = msi_field_getSize(mat->get_field());

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
    std::cout <<"[msi] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
    return M3DC1_FAILURE;
  }
  else
    if (!PCU_Comm_Self())
	std::cout <<"[msi] "<<__func__<<": matrix "<<*
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
  if (!PCU_Comm_Self()) std::cout <<"[msi ERROR] "<<__func__<<" not supported: check if amesos2 is available\n";
  return M3DC1_FAILURE;

  m3dc1_epetra* mat = m3dc1_ls::instance()->get_matrix(*matrix_id);
  if (!mat || mat->matrix_type!=M3DC1_SOLVE)
  {
    std::cout <<"[msi ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
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
    std::cout <<"[msi] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<*matrix_id<<")\n";
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
void set_adj_node_tag(pMesh m, pOwnership o, pMeshTag num_global_adj_node_tag, pMeshTag num_own_adj_node_tag)
// **********************************************
{
  int value;
  int brgType = m->getDimension()-1;

  apf::MeshEntity* e;
  apf::MeshIterator* it = m->begin(0);
  PCU_Comm_Begin();
  while ((e = m->iterate(it)))
  {
    int num_adj_node=0;
    apf::Adjacent elements;
    apf::getBridgeAdjacent(m, e, brgType, 0, elements);
    int num_adj = elements.getSize();

    for (int i=0; i<num_adj; ++i)
    {
      if (pumi_ment_isOwned(elements[i], o))
        ++num_adj_node;
    }
    m->setIntTag(e, num_own_adj_node_tag, &num_adj_node);

    if (!m->isShared(e)) continue;
    // first pass msg size to owner
    int own_partid = pumi_ment_getOwnPID(e,o);
    apf::MeshEntity* own_copy = pumi_ment_getOwnEnt(e,o);
    if (!own_copy) // own_copy does not exist so let;'s
    {

    }
    if (own_partid==PCU_Comm_Self()) continue;
    PCU_COMM_PACK(own_partid, own_copy);
    PCU_Comm_Pack(own_partid, &num_adj,sizeof(int));
  }
  m->end(it);

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

  // pass entities to owner
  std::map<apf::MeshEntity*, std::set<entMsg, classcomp> > count_map2;
  it = m->begin(0);
  PCU_Comm_Begin();
  while ((e = m->iterate(it)))
  {
    // pass entities to ownner

    std::vector<entMsg> msgs;
    apf::Adjacent elements;
    apf::getBridgeAdjacent(m, e, brgType, 0, elements);

    apf::MeshEntity* ownerEnt=pumi_ment_getOwnEnt(e,o);
    int own_partid = pumi_ment_getOwnPID(e, o);
    for(int i=0; i<elements.getSize(); ++i)
    {
      apf::MeshEntity* ownerEnt2=pumi_ment_getOwnEnt(elements[i],o);
      int owner=pumi_ment_getOwnPID(elements[i], o);
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
  m->end(it);

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
    m->setIntTag(mit->first, num_global_adj_node_tag, &num_global_adj);
  }
}
#endif
