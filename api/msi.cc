/******************************************************************************

  (c) 2017 Scientific Computation Research Center,
      Rensselaer Polytechnic Institute. All rights reserved.

  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.

*******************************************************************************/
#include "msi.h"
#include "PCU.h"
#include "apfMDS.h"
#include "msi_petsc.h"
#include "msi_solver.h"
// Added for the synchronization function
#include "apfFieldData.h"
#include "apfNew.h"
#include "apfNumbering.h"
#include "apfNumberingClass.h"
#include "apfShape.h"
using std::vector;
#include <unistd.h>
#include <cassert>
#include <iostream>
#include <vector>

// prerequisite: pumi_start();
void msi_init(int argc, char * argv[], MPI_Comm cm)
{
  msi_matrix_setComm(cm);
  PetscInitialize(&argc,&argv,NULL,NULL);
}

// declaration for use in msi_start
void set_adj_node_tag(pMesh m,
                      pOwnership,
                      pMeshTag num_global_adj_node_tag,
                      pMeshTag num_own_adj_node_tag);

void msi_start(pMesh m, pOwnership o, pShape s, MPI_Comm cm)
{
  if (o)
  {
    msi_solver::instance( )->ownership = o;
    if (!pumi_rank( ))
      std::cout << "[msi] (" << pumi_rank( ) << ") " << __func__
                << ": user-defined ownership is in use\n";
    pumi_ownership_verify(m, o);
  }
  else
  {
    msi_solver::instance( )->ownership = new apf::NormalSharing(m);
    if (!pumi_rank( ))
      std::cout << "[msi] (" << pumi_rank( ) << ") " << __func__
                << ": the default mesh ownership is in use\n";
  }
  pumi_mesh_setCount(m, o);
  if (s)
    pumi_mesh_setShape(m, s);
  else
    s = pumi_mesh_getShape(m);
  PetscMemorySetGetMaximumUsage( );
  msi_solver::instance( )->num_global_adj_node_tag =
    m->createIntTag("msi_num_global_adj_node", 1);
  msi_solver::instance( )->num_own_adj_node_tag =
    m->createIntTag("msi_num_own_adj_node", 1);
  set_adj_node_tag(m,
                   o,
                   msi_solver::instance( )->num_global_adj_node_tag,
                   msi_solver::instance( )->num_own_adj_node_tag);
  // set local numbering
  const char* name = s->getName( );
  pNumbering ln = m->findNumbering(name);
  if (!ln)
    ln = apf::numberOverlapNodes(m, name, s);
  msi_solver::instance( )->local_n = ln;
  // generate global ID's per ownership
  if (cm == MPI_COMM_NULL)
    msi_solver::instance( )->global_n =
      pumi_numbering_createGlobal(m, "pumi_global", NULL, o);
  else
    msi_solver::instance( )->global_n =
      msi_numbering_createGlobal_multiOwner(m, "pumi_global", NULL, o, cm);
  msi_solver::instance( )->vertices = new pMeshEnt[m->count(0)];
  pMeshEnt e;
  pMeshIter it = m->begin(0);
  while ((e = m->iterate(it)))
  {
#ifdef DEBUG
    assert(apf::isNumbered(msi_solver::instance( )->local_n, e, 0, 0));
#endif
    msi_solver::instance( )->vertices[msi_node_getID(e, 0)] = e;
  }
  m->end(it);
}

void msi_stop(pMesh m)
{
  apf::removeTagFromDimension(
    m, msi_solver::instance( )->num_global_adj_node_tag, 0);
  m->destroyTag(msi_solver::instance( )->num_global_adj_node_tag);
  apf::removeTagFromDimension(
    m, msi_solver::instance( )->num_own_adj_node_tag, 0);
  m->destroyTag(msi_solver::instance( )->num_own_adj_node_tag);
  pumi_numbering_delete(msi_solver::instance( )->local_n);
  pumi_numbering_delete(msi_solver::instance( )->global_n);
}

void msi_finalize()
{
  PetscFinalize();
}

// Synchronization alternative to apf::synchronizeFieldData for multiple
// ownership in parasol
template <class T>
void synchronizeFieldData_parasol(apf::FieldDataOf<T>* data,
                                  apf::Sharing* shr,
                                  MPI_Comm comm,
                                  bool delete_shr)
{
  apf::FieldBase* f = data->getField( );
  apf::Mesh* m = f->getMesh( );
  apf::FieldShape* s = f->getShape( );
  if (!shr)
  {
    shr = getSharing(m);
    delete_shr = true;
  }
  // Rank membership checking
  MPI_Group comm_group, world_group;
  int wrank[1], crank[1];
  // Group made out of this communicator and world
  MPI_Comm_group(comm, &comm_group);
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  for (int d = 0; d < 4; ++d)
  {
    if (!s->hasNodesIn(d))
      continue;
    apf::MeshEntity* e;
    apf::MeshIterator* it = m->begin(d);
    PCU_Comm_Begin( );
    while ((e = m->iterate(it)))
    {
      if ((!data->hasEntity(e)) || (!shr->isOwned(e)))
        continue;
      int n = f->countValuesOn(e);
      apf::NewArray<T> values(n);
      data->get(e, &(values[0]));
      CopyArray copies;
      shr->getCopies(e, copies);
      for (size_t i = 0; i < copies.getSize( ); ++i)
      {
        wrank[0] = copies[i].peer;
        MPI_Group_translate_ranks(world_group, 1, wrank, comm_group, crank);
        if (crank[0] != MPI_UNDEFINED)
        {
          PCU_COMM_PACK(copies[i].peer, copies[i].entity);
          PCU_Comm_Pack(copies[i].peer, &(values[0]), n * sizeof(T));
        }
      }
    }
    m->end(it);
    PCU_Comm_Send( );
    while (PCU_Comm_Receive( ))
    {
      apf::MeshEntity* e;
      PCU_COMM_UNPACK(e);
      int n = f->countValuesOn(e);
      apf::NewArray<T> values(n);
      PCU_Comm_Unpack(&(values[0]), n * sizeof(T));
      data->set(e, &(values[0]));
    }
  }
  if (delete_shr)
    delete shr;
  MPI_Group_free(&comm_group);
  MPI_Group_free(&world_group);
}
template void synchronizeFieldData_parasol<int>(apf::FieldDataOf<int>*,
                                                apf::Sharing*,
                                                MPI_Comm,
                                                bool);
template void synchronizeFieldData_parasol<double>(apf::FieldDataOf<double>*,
                                                   apf::Sharing*,
                                                   MPI_Comm,
                                                   bool);
template void synchronizeFieldData_parasol<long>(apf::FieldDataOf<long>*,
                                                 apf::Sharing*,
                                                 MPI_Comm,
                                                 bool);
void accumulateFieldData_parasol(apf::FieldDataOf<double>* data,
                                 apf::Sharing* shr,
                                 MPI_Comm comm,
                                 bool delete_shr)
{
  apf::FieldBase* f = data->getField( );
  apf::Mesh* m = f->getMesh( );
  apf::FieldShape* s = f->getShape( );
  if (!shr)
  {
    shr = getSharing(m);
    delete_shr = true;
  }
  // Rank membership checking
  MPI_Group comm_group, world_group;
  int wrank[1], crank[1];
  // Group made out of this communicator and world
  MPI_Comm_group(comm, &comm_group);
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  for (int d = 0; d < 4; ++d)
  {
    if (!s->hasNodesIn(d))
      continue;
    apf::MeshEntity* e;
    apf::MeshIterator* it = m->begin(d);
    PCU_Comm_Begin( );
    while ((e = m->iterate(it)))
    {
      if ((!data->hasEntity(e)) || m->isGhost(e) || (shr->isOwned(e)))
        continue; /* non-owners send to owners */
      apf::CopyArray copies;
      shr->getCopies(e, copies);
      int n = f->countValuesOn(e);
      apf::NewArray<double> values(n);
      data->get(e, &(values[0]));
      /* actually, non-owners send to all others,
         since apf::Sharing doesn't identify the owner */
      for (size_t i = 0; i < copies.getSize( ); ++i)
      {
        wrank[0] = copies[i].peer;  // add group removal
        MPI_Group_translate_ranks(world_group, 1, wrank, comm_group, crank);
        if (crank[0] != MPI_UNDEFINED)
        {
          PCU_COMM_PACK(copies[i].peer, copies[i].entity);
          PCU_Comm_Pack(copies[i].peer, &(values[0]), n * sizeof(double));
        }
      }
    }
    m->end(it);
    PCU_Comm_Send( );
    while (PCU_Comm_Listen( ))
      while (!PCU_Comm_Unpacked( ))
      { /* receive and add. we only care about correctness
           on the owners */
        apf::MeshEntity* e;
        PCU_COMM_UNPACK(e);
        int n = f->countValuesOn(e);
        apf::NewArray<double> values(n);
        apf::NewArray<double> inValues(n);
        PCU_Comm_Unpack(&(inValues[0]), n * sizeof(double));
        data->get(e, &(values[0]));
        for (int i = 0; i < n; ++i)
          values[i] += inValues[i];
        data->set(e, &(values[0]));
      }
  } /* broadcast back out to non-owners */
  synchronizeFieldData_parasol<double>(data, shr, comm, delete_shr);
  MPI_Group_free(&comm_group);
  MPI_Group_free(&world_group);
}
// Synchronization function for collecting field data from the same group on all
// planes Assumes that DOF ordering in apf::Field corresponds to plane ordering
// i.e. DOF i is the value on plane i; maybe just overload the previous
template <class T>
void synchronizeFieldData_parasol_all_planes(apf::FieldDataOf<T>* data,
                                             apf::Sharing* shr,
                                             int iplane,
                                             bool delete_shr)
{
  std::map<apf::MeshEntity*, std::vector<T> > all_values;
  apf::FieldBase* f = data->getField( );
  apf::Mesh* m = f->getMesh( );
  apf::FieldShape* s = f->getShape( );
  if (!shr)
  {
    shr = getSharing(m);
    delete_shr = true;
  }
  for (int d = 0; d < 4; ++d)
  {
    if (!s->hasNodesIn(d))
      continue;
    apf::MeshEntity* e;
    apf::MeshIterator* it = m->begin(d);
    PCU_Comm_Begin( );
    while ((e = m->iterate(it)))
    {
      if ((!data->hasEntity(e)) || (!shr->isOwned(e)))
        continue;
      int n = f->countValuesOn(e);
      apf::NewArray<T> values(n);
      data->get(e, &(values[0]));
      // Own plane - value
      all_values[e].resize(n);
      all_values[e][iplane] = values[iplane];
      CopyArray copies;
      shr->getCopies(e, copies);
      for (size_t i = 0; i < copies.getSize( ); ++i)
      {
        PCU_COMM_PACK(copies[i].peer, iplane);
        PCU_COMM_PACK(copies[i].peer, copies[i].entity);
        PCU_Comm_Pack(copies[i].peer, &values[iplane], sizeof(T));
      }
    }
    m->end(it);
    PCU_Comm_Send( );
    while (PCU_Comm_Receive( ))
    {
      apf::MeshEntity* e;
      int iplane = -1;
      T value;
      PCU_COMM_UNPACK(iplane);
      PCU_COMM_UNPACK(e);
      int n = f->countValuesOn(e);
      apf::NewArray<T> values(n);
      all_values[e].resize(n);  // this is far from ideal but will do for now
      PCU_Comm_Unpack(&value, sizeof(T));
      all_values[e][iplane] = value;
    }
    // If the resulting vector is not continguous, garbage will propagate
    typename std::map<apf::MeshEntity*, std::vector<T> >::iterator it_elem;
    for (it_elem = all_values.begin( ); it_elem != all_values.end( ); it_elem++)
      data->set(it_elem->first, &it_elem->second[0]);
  }
  if (delete_shr)
    delete shr;
}
template void synchronizeFieldData_parasol_all_planes<int>(
  apf::FieldDataOf<int>*,
  apf::Sharing*,
  MPI_Comm,
  bool);
template void synchronizeFieldData_parasol_all_planes<double>(
  apf::FieldDataOf<double>*,
  apf::Sharing*,
  MPI_Comm,
  bool);
template void synchronizeFieldData_parasol_all_planes<long>(
  apf::FieldDataOf<long>*,
  apf::Sharing*,
  MPI_Comm,
  bool);
// returns sequential local numbering of entity's ith node
// local numbering is based on mesh shape
int msi_node_getID(pMeshEnt e, int n)
{
  assert(apf::isNumbered(msi_solver::instance( )->local_n, e, n, 0));
  return pumi_node_getNumber(msi_solver::instance( )->local_n, e, n);
}
// returns global numbering of entity's ith node
// global numbering is based on ownership set in msi_start
int msi_node_getGlobalID(pMeshEnt e, int n)
{
  assert(apf::isNumbered(msi_solver::instance( )->global_n, e, n, 0));
  return pumi_node_getNumber(msi_solver::instance( )->global_n, e, n);
}
#define FIELDVALUELIMIT 1e100
bool value_is_nan(double val)
{
  return val != val || fabs(val) > FIELDVALUELIMIT;
}
void msi_node_setField(pField f,
                       pMeshEnt e,
                       int n,
                       int num_dof,
                       double* dof_data)
{
#ifdef DEBUG
  int scalar_type = 0;
#ifdef PETSC_USE_COMPLEX
  scalar_type = 1;
#endif
  assert(countComponents(f) == num_dof * (1 + scalar_type));
  for (int i = 0; i < num_dof * (1 + scalar_type); ++i)
    assert(!value_is_nan(dof_data[i]));
#endif
  setComponents(f, e, n, dof_data);
}
int msi_node_getField(pField f, pMeshEnt e, int n, double* dof_data)
{
  getComponents(f, e, n, dof_data);
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof /= 2;
#endif
#ifdef DEBUG
  int scalar_type = 0;
#ifdef PETSC_USE_COMPLEX
  scalar_type = 1;
#endif
  for (int i = 0; i < num_dof * (1 + scalar_type); i++)
    assert(!value_is_nan(dof_data[i]));
  int start_dof_id, end_dof_id_plus_one;
  msi_node_getFieldID(f, e, n, &start_dof_id, &end_dof_id_plus_one);
  double* data;
  msi_field_getdataptr(f, &data);
  int start = start_dof_id * (1 + scalar_type);
  for (int i = 0; i < num_dof; i++)
    assert(data[start++] == dof_data[i]);
#endif
  return num_dof;
}
//*******************************************************
void msi_node_getFieldID(pField f,
                         pMeshEnt e,
                         int n,
                         int* /* out */ start_dof_id,
                         int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof /= 2;
#endif
  int ent_id = msi_node_getID(e, n);
  *start_dof_id = ent_id * num_dof;
  *end_dof_id_plus_one = *start_dof_id + num_dof;
}
//*******************************************************
void msi_node_getGlobalFieldID(pField f,
                               pMeshEnt e,
                               int n,
                               int* /* out */ start_dof_id,
                               int* /* out */ end_dof_id_plus_one)
//*******************************************************
{
  int num_dof = apf::countComponents(f);
#ifdef PETSC_USE_COMPLEX
  num_dof /= 2;
#endif
  int ent_id = msi_node_getGlobalID(e, n);
  *start_dof_id = ent_id * num_dof;
  *end_dof_id_plus_one = *start_dof_id + num_dof;
}

pNumbering msi_numbering_createGlobal_multiOwner(pMesh m,
                                                 const char* name,
                                                 pShape s,
                                                 pOwnership o,
                                                 MPI_Comm cm)
{
  pNumbering n = m->findNumbering(name);
  if (n)
  {
    if (!pumi_rank( ))
      std::cout << "[PUMI INFO] " << __func__ << " failed: numbering \"" << name
                << "\" already exists\n";
    return n;
  }
  if (!s)
    s = m->getShape( );
  n = numberOwnedNodes(m, name, s, o);
  MPI_Comm prevComm = PCU_Get_Comm( );
  PCU_Switch_Comm(cm);
  apf::globalize(n);
  PCU_Switch_Comm(prevComm);
  synchronizeFieldData_parasol<int>(n->getData( ),
                                    o,
                                    cm,
                                    false);  // synchronize(n, o);
#ifdef DEBUG
  pMeshEnt v;
  pField f = pumi_field_create(m, "globalize", 1);
  pMeshIter it = m->begin(0);
  while ((v = m->iterate(it)))
  {
    double inum = -1;
    if (apf::isNumbered(n, v, 0, 0))
    {
      inum = pumi_node_getNumber(n, v, 0);
    }
    pumi_node_setField(f, v, 0, &inum);
  }
  m->end(it);
  pumi_mesh_write(m, "new_check_globalize", "vtk");
  MPI_Barrier(MPI_COMM_WORLD);
  pumi_field_delete(f);
#endif
  return n;
}

pOwnership msi_getOwnership( ) { return msi_solver::instance( )->ownership; }

//*******************************************************
void msi_field_assign(pField f, double* fac)
//*******************************************************
{
  int scalar_type = 0, dofPerEnt = countComponents(f);
#ifdef PETSC_USE_COMPLEX
  dofPerEnt /= 2;
  scalar_type = 1;
#endif
  std::vector<double> dofs(dofPerEnt * (1 + scalar_type), fac[0]);
  if (scalar_type)
    for (int i = 0; i < dofPerEnt; i++)
      dofs.at(2 * i + 1) = fac[1];
  pMeshEnt e;
  pMeshIter it = pumi::instance( )->mesh->begin(0);
  while ((e = pumi::instance( )->mesh->iterate(it)))
    msi_node_setField(f, e, 0, dofPerEnt, &dofs[0]);
  pumi::instance( )->mesh->end(it);
}
//*******************************************************
pField msi_field_create(pMesh m,
                        const char* /* in */ field_name,
                        int /*in*/ num_values,
                        int /*in*/ num_dofs_per_value,
                        pShape shape)
//*******************************************************
{
  int scalar_type = 0;
#ifdef PETSC_USE_COMPLEX
  scalar_type = 1;
#endif
  int components = num_values * (scalar_type + 1) * num_dofs_per_value;
  apf::Field* f = createPackedField(m, field_name, components, shape);
  msi_solver::instance( )->add_field(f, num_values);
  apf::freeze(f);  // switch dof data from tag to array
  double val[2] = {0, 0};
  msi_field_assign(f, val);
  return f;
}
//*******************************************************
int msi_field_getNumVal(pField f)
//*******************************************************
{
  return (*msi_solver::instance( )->field_container)[f];
}
//*******************************************************
int msi_field_getSize(pField f)
//*******************************************************
{
#ifdef PETSC_USE_COMPLEX
  return countComponents(f) / 2;
#else
  return countComponents(f);
#endif
}
void msi_matrix_setComm(MPI_Comm cm) { PETSC_COMM_WORLD = cm; }
/** matrix and solver functions */
//*******************************************************
msi_matrix* msi_matrix_create(int matrix_type, pField f)
//*******************************************************
{
#ifdef DEBUG
  if (!PCU_Comm_Self( ))
    std::cout << "[msi] " << __func__ << ": type " << matrix_type << ", field "
              << getName(f) << "\n";
#endif
  if (matrix_type == MSI_MULTIPLY)  // matrix for multiplication
  {
    matrix_mult* new_mat = new matrix_mult(f);
    return ( msi_matrix* )new_mat;
  }
  else
  {
    matrix_solve* new_mat = new matrix_solve(f);
    return ( msi_matrix* )new_mat;
  }
}
pField msi_matrix_getField(pMatrix mat) { return mat->get_field( ); }
//*******************************************************
void msi_matrix_assemble(pMatrix mat)
//*******************************************************
{
#ifdef DEBUG
  if (!PCU_Comm_Self( ))
    std::cout << "[msi] " << __func__ << "\n";
#endif
  mat->assemble( );
}
//*******************************************************
void msi_matrix_delete(pMatrix mat)
//*******************************************************
{
#ifdef DEBUG
  if (!PCU_Comm_Self( ))
    std::cout << "[msi] " << __func__ << "\n";
#endif
  delete mat;
}
//*******************************************************
void msi_matrix_insert(pMatrix mat,
                       int row,
                       int col,
                       int scalar_type,
                       double* val)
//*******************************************************
{
  assert(mat->get_status( ) != MSI_FIXED);
#ifdef DEBUG
  int num_values = msi_field_getNumVal(mat->get_field( ));
  int total_num_dof = msi_field_getSize(mat->get_field( ));
  int ent_id = row / total_num_dof;
  apf::MeshEntity* e = msi_solver::instance( )->vertices[ent_id];
  assert(e && !pumi::instance( )->mesh->isGhost(e));
#endif
  if (scalar_type)  // complex
    mat->set_value(row, col, INSERT_VALUES, val[0], val[1]);
  else
    mat->set_value(row, col, INSERT_VALUES, *val, 0);
}
//*******************************************************
void msi_matrix_add(pMatrix mat, int row, int col, int scalar_type, double* val)
//*******************************************************
{
  assert(mat->get_status( ) != MSI_FIXED);
#ifdef DEBUG
  int num_values = msi_field_getNumVal(mat->get_field( ));
  int total_num_dof = msi_field_getSize(mat->get_field( ));
  int ent_id = row / total_num_dof;
  apf::MeshEntity* e = msi_solver::instance( )->vertices[ent_id];
  assert(e && !pumi::instance( )->mesh->isGhost(e));
#endif
  if (scalar_type)  // complex
    mat->set_value(row, col, ADD_VALUES, val[0], val[1]);
  else
    mat->set_value(row, col, ADD_VALUES, *val, 0);
}
//*******************************************************
void msi_matrix_setBC(pMatrix mat, int row)
//*******************************************************
{
  assert(mat->get_type( ) == MSI_SOLVE);
  int num_values = msi_field_getNumVal(mat->get_field( ));
  int total_num_dof = msi_field_getSize(mat->get_field( ));
  int inode = row / total_num_dof;
  pMeshEnt e = msi_solver::instance( )->vertices[inode];
  int start_global_dof_id, end_global_dof_id_plus_one;
  msi_node_getGlobalFieldID(
    mat->get_field( ), e, 0, &start_global_dof_id, &end_global_dof_id_plus_one);
#ifdef DEBUG
  assert(!pumi::instance( )->mesh->isGhost(e));
  int start_dof_id, end_dof_id_plus_one;
  msi_node_getFieldID(
    mat->get_field( ), e, 0, &start_dof_id, &end_dof_id_plus_one);
  assert(row >= start_dof_id && row < end_dof_id_plus_one);
#endif
  int row_g = start_global_dof_id + row % total_num_dof;
  (dynamic_cast<matrix_solve*>(mat))->set_bc(row_g);
}
//*******************************************************
void msi_matrix_setLaplaceBC(pMatrix mat,
                             int row,
                             int numVals,
                             int* columns,
                             double* values)
//*******************************************************
{
  assert(mat->get_type( ) == MSI_SOLVE);
  std::vector<int> columns_g(numVals);
  int num_values = msi_field_getNumVal(mat->get_field( ));
  int total_num_dof = msi_field_getSize(mat->get_field( ));
  int inode = row / total_num_dof;
  pMeshEnt e = msi_solver::instance( )->vertices[inode];
  int start_global_dof_id, end_global_dof_id_plus_one;
  msi_node_getGlobalFieldID(
    mat->get_field( ), e, 0, &start_global_dof_id, &end_global_dof_id_plus_one);
#ifdef DEBUG
  assert(!pumi::instance( )->mesh->isGhost(e));
  int start_dof_id, end_dof_id_plus_one;
  msi_node_getFieldID(
    mat->get_field( ), e, 0, &start_dof_id, &end_dof_id_plus_one);
  assert(row >= start_dof_id && row < end_dof_id_plus_one);
  for (int i = 0; i < numVals; i++)
    assert(columns[i] >= start_dof_id && columns[i] < end_dof_id_plus_one);
#endif
  int row_g = start_global_dof_id + row % total_num_dof;
  for (int i = 0; i < numVals; i++)
  {
    columns_g.at(i) = start_global_dof_id + columns[i] % total_num_dof;
  }
  (dynamic_cast<matrix_solve*>(mat))
    ->set_row(row_g, numVals, &columns_g[0], values);
}
//*******************************************************
void msi_matrix_multiply(pMatrix mat, pField inputvec, pField outputvec)
//*******************************************************
{
#ifdef DEBUG
  if (!PCU_Comm_Self( ))
    std::cout << "[msi] " << __func__ << ": input \"" << getName(inputvec)
              << "\", output \"" << getName(outputvec) << "\"\n";
#endif
  assert(mat->get_type( ) == MSI_MULTIPLY);
  (dynamic_cast<matrix_mult*>(mat))->multiply(inputvec, outputvec);
}
//*******************************************************
void msi_matrix_solve(pMatrix mat, pField rhs, pField sol)
//*******************************************************
{
  assert(mat->get_type( ) == MSI_SOLVE);
#ifdef DEBUG
  if (!PCU_Comm_Self( ))
    std::cout << "[msi] " << __func__ << ": RHS \"" << getName(rhs)
              << "\", sol \"" << getName(sol) << "\"\n";
#endif
  (dynamic_cast<matrix_solve*>(mat))->solve(rhs, sol);
}
//*******************************************************
int msi_matrix_getNumIter(pMatrix mat)
//*******************************************************
{
  return dynamic_cast<matrix_solve*>(mat)->iterNum;
}
//*******************************************************
void msi_matrix_addBlock(pMatrix mat,
                         pMeshEnt e,
                         int rowIdx,
                         int columnIdx,
                         double* values)
//*******************************************************
{
  // need to change later, should get the value from field calls ...
  int num_values = msi_field_getNumVal(mat->get_field( ));
  int total_num_dof = msi_field_getSize(mat->get_field( ));
  int dofPerVar = total_num_dof / num_values;
  int ent_dim = 0;
  int ielm_dim = pumi::instance( )->mesh->getDimension( );
  if (pumi::instance( )->mesh->isGhost(e))
    return;
  std::vector<pMeshEnt> vertices;
  pumi_ment_getAdj(e, 0, vertices);
  int num_node = ( int )vertices.size( );
  int* nodes = new int[num_node];
  for (int i = 0; i < num_node; ++i)
    nodes[i] = msi_node_getID(vertices[i], 0);
  int start_global_dof_id, end_global_dof_id_plus_one;
  int start_global_dof, end_global_dof_id;
  // need to change later, should get the value from field calls ...
  int scalar_type = 0;
#ifdef PETSC_USE_COMPLEX
  scalar_type = 1;
#endif
  int numDofs = total_num_dof;
  int numVar = numDofs / dofPerVar;
  assert(rowIdx < numVar && columnIdx < numVar);
  int rows[1024], columns[1024];
  assert(sizeof(rows) / sizeof(int) >= dofPerVar * num_node);
  if (mat->get_type( ) == 0)  // multiply
  {
    int localFlag = 0;
    matrix_mult* mmat = dynamic_cast<matrix_mult*>(mat);
    for (int inode = 0; inode < num_node; inode++)
    {
      if (mmat->is_mat_local( ))
        msi_node_getFieldID(mat->get_field( ),
                            vertices[inode],
                            0,
                            &start_global_dof_id,
                            &end_global_dof_id_plus_one);
      else
        msi_node_getGlobalFieldID(mat->get_field( ),
                                  vertices[inode],
                                  0,
                                  &start_global_dof_id,
                                  &end_global_dof_id_plus_one);
      for (int i = 0; i < dofPerVar; i++)
      {
        rows[inode * dofPerVar + i] =
          start_global_dof_id + rowIdx * dofPerVar + i;
        columns[inode * dofPerVar + i] =
          start_global_dof_id + columnIdx * dofPerVar + i;
      }
    }
    mmat->add_values(
      dofPerVar * num_node, rows, dofPerVar * num_node, columns, values);
  }
  else  // solve
  {
    matrix_solve* smat = dynamic_cast<matrix_solve*>(mat);
    int* nodeOwner = new int[num_node];
    int* columns_bloc = new int[num_node];
    int* rows_bloc = new int[num_node];
    for (int inode = 0; inode < num_node; inode++)
    {
      nodeOwner[inode] =
        msi_solver::instance( )->ownership->getOwner(vertices[inode]);
      msi_node_getGlobalFieldID(mat->get_field( ),
                                vertices[inode],
                                0,
                                &start_global_dof_id,
                                &end_global_dof_id_plus_one);
      rows_bloc[inode] = nodes[inode] * numVar + rowIdx;
      columns_bloc[inode] = nodes[inode] * numVar + columnIdx;
      for (int i = 0; i < dofPerVar; i++)
      {
        rows[inode * dofPerVar + i] =
          start_global_dof_id + rowIdx * dofPerVar + i;
        columns[inode * dofPerVar + i] =
          start_global_dof_id + columnIdx * dofPerVar + i;
      }
    }
    int numValuesNode = dofPerVar * dofPerVar * num_node * (1 + scalar_type);
    int offset = 0;
    for (int inode = 0; inode < num_node; inode++)
    {
      if (nodeOwner[inode] != PCU_Comm_Self( ))
        smat->add_blockvalues(
          1, rows_bloc + inode, num_node, columns_bloc, values + offset);
      else
        smat->add_values(dofPerVar,
                         rows + dofPerVar * inode,
                         dofPerVar * num_node,
                         columns,
                         values + offset);
      offset += numValuesNode;
    }
    delete[] nodeOwner;
    delete[] columns_bloc;
    delete[] rows_bloc;
  }
  delete[] nodes;
}
//*******************************************************
void msi_matrix_write(pMatrix mat, const char* filename, int start_index)
//*******************************************************
{
  if (!filename)
    return msi_matrix_print(mat);
  char matrix_filename[256];
  sprintf(matrix_filename, "%s-%d", filename, PCU_Comm_Self( ));
  FILE* fp = fopen(matrix_filename, "w");
  int row, col, csize, sum_csize = 0, index = 0;
  vector<int> rows;
  vector<int> n_cols;
  vector<int> cols;
  vector<double> vals;
  mat->get_values(rows, n_cols, cols, vals);
  for (int i = 0; i < rows.size( ); ++i)
    sum_csize += n_cols[i];
  assert(vals.size( ) == sum_csize);
  fprintf(fp, "%lu\t%lu\t%lu\n", rows.size( ), n_cols.size( ), vals.size( ));
  for (int i = 0; i < rows.size( ); ++i)
  {
    row = rows[i];
    csize = n_cols[i];
    for (int j = 0; j < csize; ++j)
    {
      fprintf(fp,
              "%d\t%d\t%E\n",
              row + start_index,
              cols[index] + start_index,
              vals[index]);
      ++index;
    }
  }
  fclose(fp);
  assert(index == vals.size( ));
}
//*******************************************************
void msi_matrix_print(pMatrix mat)
//*******************************************************
{
  int row, col, csize, sum_csize = 0, index = 0;
  vector<int> rows;
  vector<int> n_cols;
  vector<int> cols;
  vector<double> vals;
  mat->get_values(rows, n_cols, cols, vals);
  for (int i = 0; i < rows.size( ); ++i)
    sum_csize += n_cols[i];
  assert(vals.size( ) == sum_csize);
  for (int i = 0; i < rows.size( ); ++i)
  {
    row = rows[i];
    csize = n_cols[i];
    for (int j = 0; j < csize; ++j)
    {
      std::cout << "[" << PCU_Comm_Self( ) << "]\t" << row << "\t"
                << cols[index] << "\t" << vals[index] << "\n";
      ++index;
    }
  }
  assert(index == vals.size( ));
}
struct entMsg
{
  int pid;
  apf::MeshEntity* ent;
  entMsg(int pid_p = 0, apf::MeshEntity* ent_p = NULL)
  {
    pid = pid_p;
    ent = ent_p;
  }
};
struct classcomp
{
  bool operator( )(const entMsg& lhs, const entMsg& rhs) const
  {
    if (lhs.ent == rhs.ent)
      return lhs.pid < rhs.pid;
    else
      return lhs.ent < rhs.ent;
  }
};
// **********************************************
void set_adj_node_tag(pMesh m,
                      pOwnership o,
                      pMeshTag num_global_adj_node_tag,
                      pMeshTag num_own_adj_node_tag)
// **********************************************
{
  int value;
  int brgType = m->getDimension( ) - 1;
  apf::MeshEntity* e;
  apf::MeshIterator* it = m->begin(0);
  PCU_Comm_Begin( );
  while ((e = m->iterate(it)))
  {
    int num_adj_node = 0;
    apf::Adjacent elements;
    apf::getBridgeAdjacent(m, e, brgType, 0, elements);
    int num_adj = elements.getSize( );
    for (int i = 0; i < num_adj; ++i)
    {
      if (pumi_ment_isOwned(elements[i], o))
        ++num_adj_node;
    }
    m->setIntTag(e, num_own_adj_node_tag, &num_adj_node);
    if (!m->isShared(e))
      continue;
    // first pass msg size to owner
    int own_partid = pumi_ment_getOwnPID(e, o);
    apf::MeshEntity* own_copy = pumi_ment_getOwnEnt(e, o);
    if (!own_copy)  // own_copy does not exist so let;'s
    {
    }
    if (own_partid == PCU_Comm_Self( ))
      continue;
    PCU_COMM_PACK(own_partid, own_copy);
    PCU_Comm_Pack(own_partid, &num_adj, sizeof(int));
  }
  m->end(it);
  PCU_Comm_Send( );
  std::map<apf::MeshEntity*, std::map<int, int> > count_map;
  while (PCU_Comm_Listen( ))
  {
    while (!PCU_Comm_Unpacked( ))
    {
      PCU_COMM_UNPACK(e);
      PCU_Comm_Unpack(&value, sizeof(int));
      count_map[e][PCU_Comm_Sender( )] = value;
    }
  }
  // pass entities to owner
  std::map<apf::MeshEntity*, std::set<entMsg, classcomp> > count_map2;
  it = m->begin(0);
  PCU_Comm_Begin( );
  while ((e = m->iterate(it)))
  {
    // pass entities to ownner
    std::vector<entMsg> msgs;
    apf::Adjacent elements;
    apf::getBridgeAdjacent(m, e, brgType, 0, elements);
    apf::MeshEntity* ownerEnt = pumi_ment_getOwnEnt(e, o);
    int own_partid = pumi_ment_getOwnPID(e, o);
    for (int i = 0; i < elements.getSize( ); ++i)
    {
      apf::MeshEntity* ownerEnt2 = pumi_ment_getOwnEnt(elements[i], o);
      int owner = pumi_ment_getOwnPID(elements[i], o);
      msgs.push_back(entMsg(owner, ownerEnt2));
      if (own_partid == PCU_Comm_Self( ))
      {
        count_map2[e].insert(*msgs.rbegin( ));
      }
    }
    if (own_partid != PCU_Comm_Self( ))
    {
      PCU_COMM_PACK(own_partid, ownerEnt);
      PCU_Comm_Pack(own_partid, &msgs.at(0), sizeof(entMsg) * msgs.size( ));
    }
  }
  m->end(it);
  PCU_Comm_Send( );
  while (PCU_Comm_Listen( ))
  {
    while (!PCU_Comm_Unpacked( ))
    {
      PCU_COMM_UNPACK(e);
      int sizeData = count_map[e][PCU_Comm_Sender( )];
      std::vector<entMsg> data(sizeData);
      PCU_Comm_Unpack(&data.at(0), sizeof(entMsg) * sizeData);
      for (int i = 0; i < data.size( ); ++i)
      {
        count_map2[e].insert(data.at(i));
      }
    }
  }
  for (std::map<apf::MeshEntity*, std::set<entMsg, classcomp> >::iterator mit =
         count_map2.begin( );
       mit != count_map2.end( );
       ++mit)
  {
    e = mit->first;
    int num_global_adj = count_map2[e].size( );
    m->setIntTag(mit->first, num_global_adj_node_tag, &num_global_adj);
  }
}
