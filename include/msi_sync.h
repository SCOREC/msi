/******************************************************************************

  (c) 2017 - 2019 Scientific Computation Research Center,
      Rensselaer Polytechnic Institute. All rights reserved.

  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.

*******************************************************************************/
#ifndef MSI_SYNC_H_
#define MSI_SYNC_H_

#include <apfField.h>
#include <apfFieldData.h>
#include <apfNumbering.h>
#include <apfNumberingClass.h>
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
