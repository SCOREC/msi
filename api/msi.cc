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

void msi_setOwnership(pOwnership o)
{  
  assert(!msi_ownership); // changing ownership is not allowed 
  msi_ownership=o;
  pumi_mesh_setCount(pumi::instance()->mesh, o);
  pumi_mesh_createGlobalID(pumi::instance()->mesh, o);
}

void msi_ment_getLocalFieldID(pMeshEnt e, pField f, int* start_dof_id, int* end_dof_id_plus_one)
{
  int num_dof = apf::countComponents(f);
#ifdef MSI_COMPLEX
  num_dof/=2;
#endif
  *start_dof_id = pumi_ment_getID(e)*num_dof;
  *end_dof_id_plus_one = *start_dof_id + num_dof;
}

void msi_ment_getGlobalFieldID(pMeshEnt e, pField f, int* start_dof_id, int* end_dof_id_plus_one)
{
  int num_dof = apf::countComponents(f);
#ifdef MSI_COMPLEX
  num_dof/=2;
#endif
  *start_dof_id = pumi_ment_getGlobalID(e)*num_dof;
  *end_dof_id_plus_one = *start_dof_id +num_dof;
}

int msi_field_getNumOwnDOF(pField f)
{
  int num_dof = apf::countComponents(f);
#ifdef MSI_COMPLEX
  num_dof/=2;
#endif
  return num_dof*pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
}

void msi_field_getOwnDOFID(pField f, 
    int* /* out */ start_dof_id, int* /* out */ end_dof_id_plus_one)
{
  int num_dof = apf::countComponents(f);
#ifdef MSI_COMPLEX
  num_dof/=2;
#endif
  int num_own_ent = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
  int start_id = num_own_ent;
  PCU_Exscan_Ints(&start_id,1);

  *start_dof_id=start_id*num_dof;
  *end_dof_id_plus_one=*start_dof_id+num_own_ent*num_dof;
}

#ifdef MSI_PETSC
/** matrix and solver functions */
std::map<int, int> matHit;
int getMatHit(int id) { return matHit[id];};
void addMatHit(int id) { matHit[id]++; }

//*******************************************************
void msi_matrix_create(int matrix_id, int matrix_type, pField f)
//*******************************************************
{  
  static bool set_ownership=false;
  if (!msi_ownership && !set_ownership)
  {
    if (!pumi_rank())
      std::cout<<"[MSI INFO] "<<__func__<<": the mesh ownership is set to the PUMI default\n";
    pumi_mesh_setCount(pumi::instance()->mesh);
    pumi_mesh_createGlobalID(pumi::instance()->mesh);
  }

  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(!mat);

#ifdef DEBUG
  if (!PCU_Comm_Self())
    std::cout<<"[MSI INFO] "<<__func__<<": ID "<<matrix_id<<", field "<<getName(f)<<"\n";
#endif 

  if (matrix_type==MSI_MULTIPLY) // matrix for multiplication
  {
    matrix_mult* new_mat = new matrix_mult(matrix_id, f);
    msi_solver::instance()->add_matrix(matrix_id, (msi_matrix*)new_mat);
  }
  else 
  {
    matrix_solve* new_mat= new matrix_solve(matrix_id, f);
    msi_solver::instance()->add_matrix(matrix_id, (msi_matrix*)new_mat);
  }
}

//*******************************************************
void msi_matrix_freeze(int matrix_id) 
//*******************************************************
{
  double t1 = MPI_Wtime();
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert (mat);
  mat->assemble();
}

//*******************************************************
void msi_matrix_delete(int matrix_id)
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat);

#ifdef DEBUG
  if (!PCU_Comm_Self())
    std::cout<<"[MSI INFO] "<<__func__<<": ID "<<matrix_id<<"\n";
#endif

  typedef std::map<int, msi_matrix*> matrix_container_map;
  msi_solver::instance()->matrix_container->erase(matrix_container_map::key_type(matrix_id));
  delete mat;
}

//*******************************************************
void msi_matrix_insert(int matrix_id, int row, 
         int col, double* val)
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat);

  if (mat->get_status()==MSI_FIXED)
  {
    if (!PCU_Comm_Self())
      std::cout <<"[MSI ERROR] "<<__func__<<" failed: matrix with id "<<matrix_id<<" is fixed\n";
    return;
  }

#ifdef DEBUG
  pField field = mat->get_field();
  int total_num_dof = apf::countComponents(field);
#ifdef MSI_COMPLEX
  total_num_dof/=2;
#endif
  int num_values = mat->get_num_field_value();
  int ent_id = row/total_num_dof;
  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, 0, ent_id);
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));
#endif

#ifdef MSI_COMPLEX
    mat->set_value(row, col, INSERT_VALUES, val[0], val[1]);
#else
    mat->set_value(row, col, INSERT_VALUES, *val, 0);
#endif
}

//*******************************************************
void msi_matrix_add (int matrix_id, int row, int col, double* val) //globalinsertval_
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat);

  if (mat->get_status()==MSI_FIXED)
  {
    if (!PCU_Comm_Self())
      std::cout <<"[MSI ERROR] "<<__func__<<" failed: matrix with id "<<matrix_id<<" is fixed\n";
    return;
  }

#ifdef DEBUG
  pField field = mat->get_field();
  int total_num_dof = apf::countComponents(field);
#ifdef MSI_COMPLEX
  total_num_dof/=2;
#endif
  int num_values = mat->get_num_field_value();

  int ent_id = row/total_num_dof;
  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, 0, ent_id);
  assert(e && !pumi::instance()->mesh->isGhost(e));
#endif

#ifdef MSI_COMPLEX
    mat->set_value(row, col, ADD_VALUES, val[0], val[1]);
#else
    mat->set_value(row, col, ADD_VALUES, *val, 0);
#endif
}

//*******************************************************
void msi_matrix_addBlock(int matrix_id, pMeshEnt e, 
          int rowIdx, int columnIdx, double* values)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat);

  pField field = mat->get_field();
  int total_num_dof = apf::countComponents(field);
#ifdef MSI_COMPLEX
  total_num_dof/=2;
#endif
  int num_values = mat->get_num_field_value();
  int dofPerVar=total_num_dof/num_values;

  int ielm_dim = pumi::instance()->mesh->getDimension();

  if (pumi::instance()->mesh->isGhost(e)) return;
  std::vector<pMeshEnt> adj_nodes;
  pumi_ment_getAdj(e, 0, adj_nodes);
  int nodes_per_element = adj_nodes.size();

  int* nodes = new int[nodes_per_element];
  for (int i=0; i<nodes_per_element; ++i)
    nodes[i] = getMdsIndex(pumi::instance()->mesh, adj_nodes[i]);

  int start_global_dof_id,end_global_dof_id_plus_one;
  // need to change later, should get the value from field calls ...

  int scalar_type = 0;
#ifdef MSI_COMPLEX
  scalar_type=1;
#endif

  int numDofs = total_num_dof;
  int numVar = numDofs/dofPerVar;
  assert(rowIdx<numVar && columnIdx<numVar);

  int rows[1024], columns[1024];
  assert(sizeof(rows)/sizeof(int)>=dofPerVar*nodes_per_element);

  if(mat->get_type()==MSI_MULTIPLY)
  {
    int localFlag=0;
    matrix_mult* mmat = dynamic_cast<matrix_mult*> (mat);
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      pMeshEnt e = pumi_mesh_findEnt(pumi::instance()->mesh, 0, nodes[inode]);
      if (mmat->is_mat_local()) 
        msi_ment_getLocalFieldID (e, field, &start_global_dof_id, &end_global_dof_id_plus_one);
      else 
        msi_ment_getGlobalFieldID (e, field, &start_global_dof_id, &end_global_dof_id_plus_one);
      for(int i=0; i<dofPerVar; i++)
      {
        rows[inode*dofPerVar+i]=start_global_dof_id+rowIdx*dofPerVar+i;
        columns[inode*dofPerVar+i]=start_global_dof_id+columnIdx*dofPerVar+i;
      }
    }
    mmat->add_values(dofPerVar*nodes_per_element, rows, dofPerVar*nodes_per_element, columns, values);
  }
  else
  {
    matrix_solve* smat = dynamic_cast<matrix_solve*> (mat);
    int nodeOwner[6];
    int columns_bloc[6], rows_bloc[6];
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      pMeshEnt e = pumi_mesh_findEnt(pumi::instance()->mesh, 0, nodes[inode]);
      nodeOwner[inode] = pumi_ment_getOwnPID(e);
      msi_ment_getGlobalFieldID(e, field, &start_global_dof_id, &end_global_dof_id_plus_one);
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
  delete [] nodes;
}


//*******************************************************
void msi_matrix_setbc(int matrix_id, int row)
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat);

  if (mat->get_type()!=MSI_SOLVE)
  { 
    if (!PCU_Comm_Self())
      std::cout <<"[MSI ERROR] "<<__func__<<" not supported with matrix for multiplication (id"<<matrix_id<<")\n";
    return;
  }
  pField field = mat->get_field();
  int total_num_dof = apf::countComponents(field);
#ifdef MSI_COMPLEX
  total_num_dof/=2;
#endif
  int inode = row/total_num_dof;
  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, 0, inode);

  int start_global_dof_id, end_global_dof_id_plus_one;
  msi_ment_getGlobalFieldID(e, field, &start_global_dof_id, &end_global_dof_id_plus_one);

#ifdef DEBUG
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));
  int start_dof_id, end_dof_id_plus_one;
  msi_ment_getLocalFieldID(e, field, &start_dof_id, &end_dof_id_plus_one);
  assert(row>=start_dof_id&&row<end_dof_id_plus_one);
#endif
  int row_g = start_global_dof_id+row%total_num_dof;
  (dynamic_cast<matrix_solve*>(mat))->set_bc(row_g);
}

//*******************************************************
void msi_matrix_setlaplacebc(int matrix_id, int row,
         int numVals, int* columns, double* values)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat && mat->get_type()==MSI_SOLVE);

  std::vector <int> columns_g(numVals);
  pField field = mat->get_field();
  int total_num_dof = apf::countComponents(field);
#ifdef MSI_COMPLEX
  total_num_dof/=2;
#endif

  int inode = row/total_num_dof;
  apf::MeshEntity* e =getMdsEntity(pumi::instance()->mesh, 0, inode);

  int start_global_dof_id, end_global_dof_id_plus_one;
  msi_ment_getGlobalFieldID(e, field, &start_global_dof_id, &end_global_dof_id_plus_one);

#ifdef DEBUG
  assert(e);
  assert(!pumi::instance()->mesh->isGhost(e));

  int start_dof_id, end_dof_id_plus_one;
  msi_ment_getLocalFieldID(e, field, &start_dof_id, &end_dof_id_plus_one);
  assert(row>=start_dof_id&&row<end_dof_id_plus_one);
  for (int i=0; i<numVals; i++)
    assert(columns[i]>=start_dof_id&&columns[i]<end_dof_id_plus_one);
#endif

  int row_g = start_global_dof_id+row%total_num_dof;
  for(int i=0; i<numVals; ++i)
    columns_g.at(i) = start_global_dof_id+columns[i]%total_num_dof;
  (dynamic_cast<matrix_solve*>(mat))->set_row(row_g, numVals, &columns_g[0], values);
}

void msi_matrix_solve(int matrix_id, pField x, pField rhs_sol) //solveSysEqu_
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert (mat && mat->get_type()==MSI_SOLVE);

#ifdef DEBUG
  if (!PCU_Comm_Self())
     std::cout <<"[MSI INFO] "<<__func__<<": matrix "<<matrix_id<<", field "<<rhs_sol<<"\n";
#endif

  (dynamic_cast<matrix_solve*>(mat))->solve(x, rhs_sol);
  addMatHit(matrix_id);
}

//*******************************************************
void msi_matrix_multiply(int matrix_id, pField invec, pField outvec) 
//*******************************************************
{  
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert (mat && mat->get_type()==MSI_MULTIPLY);

  (dynamic_cast<matrix_mult*>(mat))->multiply(invec, outvec);
  addMatHit(matrix_id);
}

//*******************************************************
void msi_matrix_flush(int matrix_id)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat);
  mat->flushAssembly();
}

//*******************************************************
int msi_matrix_getiternum(int matrix_id)
//*******************************************************
{ 
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat);
  return dynamic_cast<matrix_solve*> (mat)->iterNum;
}


//*******************************************************
void msi_matrix_write(int matrix_id, const char* filename, int start_index)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat);

  if (!filename) 
  { 
    msi_matrix_print(matrix_id);
    return;
  }

  char matrix_filename[256];
  sprintf(matrix_filename,"%s-%d",filename, PCU_Comm_Self());
  FILE * fp =fopen(matrix_filename, "w");

  int row, col, csize, sum_csize=0, index=0;

  std::vector<int> rows;
  std::vector<int> n_cols;
  std::vector<int> cols;
  std::vector<double> vals;

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
void msi_matrix_print(int matrix_id)
//*******************************************************
{
  msi_matrix* mat = msi_solver::instance()->get_matrix(matrix_id);
  assert(mat);

  int row, col, csize, sum_csize=0, index=0;

  std::vector<int> rows;
  std::vector<int> n_cols;
  std::vector<int> cols;
  std::vector<double> vals;

  mat->get_values(rows, n_cols, cols, vals);
  for (int i=0; i<rows.size(); ++i)
    sum_csize += n_cols[i];
  assert(vals.size()==sum_csize);

  if (!PCU_Comm_Self()) 
    std::cout<<"[MSI INFO] "<<__func__<<": printing matrix "<<matrix_id<<"\n";

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
  assert(apf::countComponents(f)*pumi::instance()->mesh->count(0)==x->MyLength());

  for(int i=0; i<x->MyLength(); ++i)
  { 
    assert(!value_is_nan((*x)[0][i]) && !value_is_nan(field_data[i]));

    if (!(msi_double_isequal((*x)[0][i], field_data[i])))
      std::cout<<"[p"<<PCU_Comm_Self()<<"] x["<<i<<"]="<<(*x)[0][i]
                <<", field_data["<<i<<"]="<<field_data[i]<<"\n";
      assert(msi_double_isequal((*x)[0][i], field_data[i]));
  }
}

void msi_epetra_create(int matrix_id, int matrix_type, int field_id)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  assert(mat);

  // check field exists
  if (!pumi::instance()->field_container || !pumi::instance()->field_container->count(f))
  {
    if (!PCU_Comm_Self())
      std::cout <<"[MSI ERROR] "<<__func__<<" failed: field with id "<<f<<" doesn't exist\n";
    return; 
  }
  int scalar_type=0;
#ifdef MSI_COMPLEX
  scalar_type=1;
#endif
  msi_ls::instance()->add_matrix(matrix_id, new msi_epetra(matrix_id, matrix_type, scalar_type, f));
}

void msi_epetra_delete(int matrix_id)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  assert(mat);

  typedef std::map<int, msi_epetra*> matrix_container_map;
  msi_ls::instance()->matrix_container->erase(matrix_container_map::key_type(matrix_id));
  mat->destroy();
  delete mat;
}

void msi_epetra_insert(int matrix_id, int row, int col, double* val)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  assert (mat);

  int err = mat->epetra_mat->ReplaceGlobalValues(*row, 1, val, col);
  if (err) {
    err =mat->epetra_mat->InsertGlobalValues(*row, 1, val, col);
    assert(err == 0);
  }
}

void print_elem (int elem_id)
{

  apf::Mesh2* m=pumi::instance()->mesh;
  
  int num_node_per_element;
  apf::Downward downward;

  int ielm_dim = pumi::instance()->mesh->getDimension();
  apf::MeshEntity* e = getMdsEntity(m, ielm_dim, elem_id);
  num_node_per_element = m->getDownward(e, 0, downward);
 
  int *id = new int[num_node_per_element];
  for (int i=0; i<num_node_per_element; ++i)
    id[i] = pumi_ment_getGlobalID(downward[i]);

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

// seol -- this does weird thing so shouldn't be used
// equivalent to Petsc::MatSetValues(*A, rsize, rows, csize, columns, &petscValues[0], ADD_VALUES);
void epetra_add_values_wrong(Epetra_CrsMatrix* mat, int rsize, int * rows, int csize, int * columns, double* values)
{
  assert(!mat->IndicesAreLocal() && !mat->IndicesAreContiguous());

  for (int i=0; i<rsize; i++)
  {
    int ierr = mat->SumIntoGlobalValues(rows[i], csize, &values[i*csize], columns);
    if (ierr) 
      ierr =mat->InsertGlobalValues(rows[i], csize, &values[i*csize], columns);
    assert(!ierr);
  } // for i
}

void msi_epetra_addblock(int matrix_id, pMeshEnt e, int rowVarIdx, int * columnVarIdx, double * values)
{

  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  assert(mat);

  pField field = mat->get_field();
  int total_num_dof = apf::countComponents(f);
#ifdef MSI_COMPLEX
  total_num_dof/=2;
#endif
  int num_values = mat->get_num_field_value();
  int dofPerVar=total_num_dof/num_values;

  std::vector<pMeshEnt>& nodes;
  pumi_ment_getAdj(e, 0, nodes);
  int nodes_per_element = nodes.size();

  int start_global_dof_id,end_global_dof_id_plus_one;

  // need to change later, should get the value from field calls ...
  int scalar_type = 0;
#ifdef MSI_COMPLEX
    scalar_type=1;
#endif
  int numDofs = total_num_dof;
  int numVar = numDofs/dofPerVar;
  assert(*rowVarIdx<numVar && *columnVarIdx<numVar);
  int rows = new int[dofPerVar*nodes_per_element];
  int columns = new int[dofPerVar*nodes_per_element];

  if (mat->matrix_type==MSI_MULTIPLY)
  {
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      pMeshEnt e = pumi_mesh_findEnt(pumi::instance()->mesh, 0, nodes[inode]);
      msi_ment_getGlobalFieldID(e, &field, &start_global_dof_id, &end_global_dof_id_plus_one);
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
  else //MSI_SOLVE
  {
    int nodeOwner[6];
    int columns_bloc[6], rows_bloc[6];
    for(int inode=0; inode<nodes_per_element; inode++)
    {
      pMeshEnt e = pumi_mesh_findEnt(pumi::instance()->mesh, 0, nodes[inode]);
      nodeOwner[inode] = pumi_ment_getOwnPID(e);
      msi_ment_getGlobalFieldID(e, field, &start_global_dof_id, &end_global_dof_id_plus_one);
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
}

void msi_epetra_setbc(int matrix_id, int row)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  if (!mat || mat->matrix_type!=MSI_SOLVE)
  {
    std::cout <<"[MSI ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<matrix_id<<")\n";
    return;
  }

  pField field = mat->get_field();
  int total_num_dof = apf::countComponents(f);
#ifdef MSI_COMPLEX
  total_num_dof/=2;
#endif
  int inode = *row/total_num_dof;
  int start_global_dof_id, end_global_dof_id_plus_one;
  pMeshEnt e = pumi_mesh_findEnt(pumi::instance()->mesh, 0, inode);

  msi_ment_getGlobalFieldID(e, field, &start_global_dof_id, &end_global_dof_id_plus_one);
#ifdef DEBUG
  int start_dof_id, end_dof_id_plus_one;
  msi_ment_getLocalFieldID(e, field, &start_dof_id, &end_dof_id_plus_one);
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
}

void msi_epetra_setlaplacebc (int * matrix_id, int *row, int * numVals, int *columns, double * values)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  if (!mat || mat->matrix_type!=MSI_SOLVE)
  {
    std::cout <<"[MSI ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<matrix_id<<")\n";
    return;
  }

  std::vector <global_ordinal_type> columns_g(*numVals);
  pField field = mat->get_field();
  int total_num_dof = apf::countComponents(f);
#ifdef MSI_COMPLEX
  total_num_dof/=2;
#endif
  int inode = *row/total_num_dof;

  int start_global_dof_id, end_global_dof_id_plus_one;
  pMeshEnt e = pumi_mesh_findEnt(pumi::instance()->mesh, 0, inode);
  msi_ment_getGlobalFieldID(e, field, &start_global_dof_id, &end_global_dof_id_plus_one);

#ifdef DEBUG
  int start_dof_id, end_dof_id_plus_one;
  msi_ment_getLocalFieldID(e, field, &start_dof_id, &end_dof_id_plus_one);
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
}

void msi_epetra_print(int matrix_id)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  assert(mat);

  // assemble matrix
  Epetra_Export exporter(/*target*/*(mat->_overlap_map),/*source*/*(mat->_owned_map));
  Epetra_CrsMatrix A(Copy, *(mat->_owned_map), mat->nge);
  A.Export(*(mat->epetra_mat),exporter,Add);
  A.FillComplete();
  A.OptimizeStorage();
  A.MakeDataContiguous();
  A.Print(cout);
}

void msi_epetra_write(int matrix_id, const char* filename, int skip_zero, int start_index)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  assert(mat);

  if (!filename)
  {
    msi_epetra_print(matrix_id);
    return;
  }

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
}

void copyEpetraVec2Field(Epetra_MultiVector* x, apf::Field* f)
{
  int start_global_dofid, num_dof = apf::countComponents(f);
  std::vector<double> dof_data(num_dof);
  apf::Mesh2* m = pumi::instance()->mesh;
  apf::MeshEntity* e;

  int index=0;
  apf::MeshIterator* it = m->begin(0);
  while ((e = m->iterate(it)))
  {
    if (pumi_ment_getOwnPID(e)!=PCU_Comm_Self()) continue;
    for(int i=0; i<num_dof; ++i)
      dof_data.at(i)=(*x)[0][index++];
    setComponents(f, e, 0, &(dof_data[0]));
  }
  m->end(it);
  assert(index == num_dof*pumi::instance()->num_own_ent[0]);
  synchronize_field(f);
}

void msi_solver_aztec_old(int matrix_id, int x_int, int b_int, int num_iter, 
  double* tolerance,const char* krylov_solver, const char* preconditioner)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  if (!mat || mat->matrix_type!=MSI_SOLVE)
  {
    std::cout <<"[MSI ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<matrix_id<<")\n";
    return;
  }
  else
    if (!PCU_Comm_Self())
      std::cout <<"[MSI INFO] "<<__func__<<": matrix "<<* matrix_id<<", field "<<* x_int<<" (tol "<<*tolerance<<")\n";

  // assemble matrix
  Epetra_Export exporter(/*target*/*(mat->_overlap_map),
			 /*source*/*(mat->_owned_map));
  Epetra_CrsMatrix A(Copy, *(mat->_owned_map), mat->nge);
  A.Export(*(mat->epetra_mat),exporter,Add);
  A.FillComplete();

//  const char* mfilename = mfilename_str.c_str();
//  EpetraExt::RowMatrixToMatlabFile(mfilename, A);

  // copy field to vec  
  apf::Field* b_field = (*(pumi::instance()->field_container))[*b_int]->get_field();
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
  solver.SetAztecOption(AZ_solver, AZ_gmres);
  solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
  solver.SetAztecOption(AZ_subdomain_solve, AZ_ilu);
  int overlap = 1;
  solver.SetAztecOption(AZ_overlap, overlap);

  solver.Iterate(*num_iter,*tolerance);
  mat->num_solver_iter = solver.NumIters();
  
  apf::Field* x_field = (*(pumi::instance()->field_container))[*x_int]->get_field();
  copyEpetraVec2Field(&x, x_field);
}

// For filtering spaces and control characters in Trilinos
// options
bool invalidChar (char c) 
{  
  return !((c > 65 && c < 90) ||
	   (c > 97 && c < 122) ||
	   (c == '_'));
} 


void msi_solver_aztec(int matrix_id, int x_int, int
		       b_int, int num_iter, double* tolerance,
		       const char* krylov_solver, const char*
		       preconditioner, const char* sub_dom_solver,
		       int overlap, int graph_fill, double*
		       ilu_drop_tol, double* ilu_fill, double*
		       ilu_omega, int poly_ord)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  if (!mat || mat->matrix_type!=MSI_SOLVE)
  {
    std::cout <<"[MSI ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<matrix_id<<")\n";
    return;
  }
  else
    if (!PCU_Comm_Self())
	std::cout <<"[MSI INFO] "<<__func__<<": matrix "<<*
	matrix_id<<", field "<<* x_int<<" (tol "<<*tolerance<<")\n";

  // assemble matrix
  Epetra_Export exporter(/*target*/*(mat->_overlap_map),
			 /*source*/*(mat->_owned_map));
  Epetra_CrsMatrix A(Copy, *(mat->_owned_map), mat->nge);
  A.Export(*(mat->epetra_mat),exporter,Add);
  A.FillComplete();
  A.OptimizeStorage();
  A.MakeDataContiguous();

  // copy field to vec  
  apf::Field* b_field = (*(pumi::instance()->field_container))[*b_int]->get_field();
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
  
  apf::Field* x_field = (*(pumi::instance()->field_container))[*x_int]->get_field();
  copyEpetraVec2Field(&x, x_field);
  return;
}

int msi_solver_getnumiter(int matrix_id)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  return mat->num_solver_iter;
}

// local matrix multiplication
// do accumulate(out_field) for global result
void msi_epetra_multiply(int matrix_id, int in_int, int out_int)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  if (!mat || mat->matrix_type!=MSI_MULTIPLY)
  {
    std::cout <<"[MSI ERROR] "<<__func__<<" matrix not exists or matrix type mismatch (id"<<matrix_id<<")\n";
    return;
  }

  if (!mat->epetra_mat->Filled())
    mat->epetra_mat->FillComplete();
  assert(mat->epetra_mat->Filled());

  apf::Field* in_field = (*(pumi::instance()->field_container))[*in_int]->get_field();
  double* in_field_data =getArrayData(in_field);

  Epetra_MultiVector x(mat->epetra_mat->RowMap(), 1);
  // copy field to vec
  for (int i=0; i<x.MyLength(); ++i)
    x[0][i] = in_field_data[i];
  Epetra_MultiVector b(mat->epetra_mat->RowMap(), 1);
  EPETRA_CHK_ERR(mat->epetra_mat->Multiply(false, x, b));
  apf::Field* out_field = (*(pumi::instance()->field_container))[*out_int]->get_field();
  double* out_field_data =getArrayData(out_field);
  b.ExtractCopy(out_field_data, b.MyLength());
  accumulate(out_field);
}

void msi_epetra_freeze(int matrix_id)
{
  msi_epetra* mat = msi_ls::instance()->get_matrix(matrix_id);
  assert(mat);
  mat->epetra_mat->FillComplete();
  assert(mat->epetra_mat->Filled());
}
#endif
