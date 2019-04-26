#ifndef MSI_SYNC_H_
#define MSI_SYNC_H_

#include <apfFieldData.h>
#include <mpi.h>

template <class T>
void synchronizeFieldData_parasol(apf::FieldDataOf<T>* data,
                                  apf::Sharing* shr,
                                  MPI_Comm comm,
                                  bool delete_shr);
void accumulateFieldData_parasol(apf::FieldDataOf<double>* data,
                                 apf::Sharing* shr,
                                 MPI_Comm comm,
                                 bool delete_shr);
template <class T>
void synchronizeFieldData_parasol_all_planes(apf::FieldDataOf<T>* data,
                                             apf::Sharing* shr,
                                             int iplane,
                                             bool delete_shr);

#endif
