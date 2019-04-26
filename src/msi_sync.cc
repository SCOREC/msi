/******************************************************************************

  (c) 2017 - 2019 Scientific Computation Research Center,
      Rensselaer Polytechnic Institute. All rights reserved.

  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.

*******************************************************************************/
#include "msi_sync.h"
#include <PCU.h>
#include <pumi.h>

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

