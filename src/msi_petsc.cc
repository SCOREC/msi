/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifdef MSI_PETSC
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

using std::vector;
using std::set;

void printMemStat()
{
  PetscLogDouble mem, mem_max;
  PetscMemoryGetCurrentUsage(&mem);
  PetscMemoryGetMaximumUsage(&mem_max);
  std::cout<<"\tMemory usage (MB) reported by PetscMemoryGetCurrentUsage: Rank "<<PCU_Comm_Self()<<" current "<<mem/1e6<<std::endl;
}

int matrix_solve::initialize()
{
  // initialize matrix
  setupMat();
  preAllocate();
  setUpRemoteAStruct();
  int ierr = MatSetUp (*A); // "MatSetUp" sets up internal matrix data structure for the later use
  //disable error when preallocate not enough
  //check later
  ierr = MatSetOption(*A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE); CHKERRQ(ierr);
  //ierr = MatSetOption(*A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE); CHKERRQ(ierr);
  ierr = MatSetOption(*A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE); CHKERRQ(ierr);
  CHKERRQ(ierr);
}

int matrix_mult::initialize()
{
  // initialize matrix
  setupMat();
  preAllocate();
  int ierr = MatSetUp (*A); // "MatSetUp" sets up internal matrix data structure for the later use
  //disable error when preallocate not enough
  //check later
  ierr = MatSetOption(*A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE); CHKERRQ(ierr);
  ierr = MatSetOption(*A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE); CHKERRQ(ierr);
  CHKERRQ(ierr);
}


msi_matrix::msi_matrix(pField f): field(f)
{
  mat_status = MSI_NOT_FIXED;
  A=new Mat;
}

int msi_matrix::destroy()
{
  PetscErrorCode ierr = MatDestroy(A);
  CHKERRQ(ierr);    
}

msi_matrix::~msi_matrix()
{
  destroy();
  delete A;
} 

int msi_matrix::set_value(int row, int col, int operation, double real_val, double imag_val) //insertion/addition with global numbering
{
  if (mat_status == MSI_FIXED)
    return MSI_FAILURE;
  PetscErrorCode ierr;

#ifndef PETSC_USE_COMPLEX
  if (operation)
    ierr = MatSetValue(*A, row, col, real_val, ADD_VALUES);
  else
    ierr = MatSetValue(*A, row, col, real_val, INSERT_VALUES);
#else // #ifdef PETSC_USE_COMPLEX
  PetscScalar value = complex<double>(real_val,imag_val);
  if (operation)
    ierr = MatSetValue(*A, row, col, value, ADD_VALUES);
  else
    ierr = MatSetValue(*A, row, col, value, INSERT_VALUES);
#endif
  CHKERRQ(ierr);
}

int msi_matrix::add_values(int rsize, int * rows, int csize, int * columns, double* values)
{
  if (mat_status == MSI_FIXED)
    return MSI_FAILURE;
  PetscErrorCode ierr;
#if defined(DEBUG) || defined(PETSC_USE_COMPLEX)
  vector<PetscScalar> petscValues(rsize*csize);
  for(int i=0; i<rsize; i++)
  {
    for(int j=0; j<csize; j++)
    {
#ifndef PETSC_USE_COMPLEX
      petscValues.at(i*csize+j)=values[i*csize+j];
#else 
      petscValues.at(i*csize+j)=complex<double>(values[2*i*csize+2*j], values[2*i*csize+2*j+1]);
#endif
    }
  }

  ierr = MatSetValues(*A, rsize, rows, csize, columns, &petscValues[0], ADD_VALUES);
#else
  ierr = MatSetValues(*A, rsize, rows, csize, columns, (PetscScalar*)values, ADD_VALUES);
#endif
  CHKERRQ(ierr);
}
int matrix_solve::add_blockvalues(int rbsize, int * rows, int cbsize, int * columns, double* values)
{
  int bs;
  MatGetBlockSize(remoteA, &bs);
  vector<PetscScalar> petscValues(rbsize*cbsize*bs*bs);

  for(int i=0; i<rbsize*bs; i++)
  {
    for(int j=0; j<cbsize*bs; j++)
    {
#ifndef PETSC_USE_COMPLEX
      petscValues.at(i*cbsize*bs+j)=values[i*cbsize*bs+j];
#else
      petscValues.at(i*cbsize*bs+j)=complex<double>(values[2*i*cbsize*bs+2*j], values[2*i*cbsize*bs+2*j+1]);
#endif
    }
  }
  int ierr = MatSetValuesBlocked(remoteA,rbsize, rows, cbsize, columns, &petscValues[0], ADD_VALUES);
}

int msi_matrix::get_values(vector<int>& rows, vector<int>& n_columns, vector<int>& columns, vector<double>& values)
{
  if (mat_status != MSI_FIXED)
    return MSI_FAILURE;
#ifdef PETSC_USE_COMPLEX
   if (!PCU_Comm_Self())
     std::cout<<"[MSI ERROR] "<<__func__<<": not supported for complex\n";
   return MSI_FAILURE;
#else
  PetscErrorCode ierr;
  PetscInt rstart, rend, ncols;
  const PetscInt *cols;
  const PetscScalar *vals;

  ierr = MatGetOwnershipRange(*A, &rstart, &rend);
  CHKERRQ(ierr);
  for (PetscInt row=rstart; row<rend; ++row)
  { 
    ierr = MatGetRow(*A, row, &ncols, &cols, &vals);
    CHKERRQ(ierr);
    rows.push_back(row);
    n_columns.push_back(ncols);
    for (int i=0; i<ncols; ++i)
    {  
      columns.push_back(cols[i]);
      values.push_back(vals[i]);
    }
    ierr = MatRestoreRow(*A, row, &ncols, &cols, &vals);
    CHKERRQ(ierr);
  }
  assert(rows.size()==rend-rstart);
  return MSI_SUCCESS;
#endif
}


// ***********************************
// 		matrix_mult
// ***********************************

int matrix_mult::multiply(pField in_field, pField out_field)
{
  if(!localMat)
  {
    Vec b, c;
    copyField2PetscVec(in_field, b);
    int ierr = VecDuplicate(b, &c);CHKERRQ(ierr);
    MatMult(*A, b, c);
    copyPetscVec2Field(c, out_field);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = VecDestroy(&c); CHKERRQ(ierr);
    return 0;
  }
  else
  {
    Vec b, c;
    int num_vtx=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);
    int num_dof = num_vtx*apf::countComponents(in_field);
#ifdef  PETSC_USE_COMPLEX 
    num_dof /= 2;
#endif

#ifdef DEBUG
    int num_dof2 = num_vtx*apf::countComponents(out_field);
#ifdef  PETSC_USE_COMPLEX 
    num_dof2 /= 2;
#endif
    assert(num_dof==num_dof2);
#endif
    int bs;
    int ierr;
    MatGetBlockSize(*A, &bs);
    PetscScalar * array[2];
    msi_field_getdataptr(in_field, (double**)array);

    ierr = VecCreateSeqWithArray( PETSC_COMM_SELF, bs, num_dof, (PetscScalar*) array[0],&b); CHKERRQ(ierr);
    msi_field_getdataptr(out_field, (double**)array+1);

    ierr = VecCreateSeqWithArray( PETSC_COMM_SELF, bs, num_dof, (PetscScalar*) array[1],&c); CHKERRQ(ierr);
    ierr=VecAssemblyBegin(b);  CHKERRQ(ierr);
    ierr=VecAssemblyEnd(b);  CHKERRQ(ierr);
    ierr=VecAssemblyBegin(c);  CHKERRQ(ierr);
    ierr=VecAssemblyEnd(c);  CHKERRQ(ierr);
    MatMult(*A, b, c);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = VecDestroy(&c); CHKERRQ(ierr);

    pumi_field_accumulate(out_field);
  }
}
int matrix_mult::assemble()
{
  PetscErrorCode ierr;
  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY); 
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  set_status(MSI_FIXED);
}

// ***********************************
// 		matrix_solve
// ***********************************
matrix_solve::matrix_solve(pField f): msi_matrix(f) 
{  
  ksp = new KSP;
  kspSet=0;
  initialize();
}

matrix_solve::~matrix_solve()
{
  if(kspSet)
    KSPDestroy(ksp);
  delete ksp;
  if(mat_status==MSI_NOT_FIXED) MatDestroy(&remoteA);
}

int matrix_solve::assemble()
{
  PetscErrorCode ierr;
  double t1 = MPI_Wtime(), t2=t1;

    ierr = MatAssemblyBegin(remoteA, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(remoteA, MAT_FINAL_ASSEMBLY);
    t2 = MPI_Wtime();
    //pass remoteA to ownnering process
    int vertex_type=0, brgType = 2;
    if (pumi::instance()->mesh->getDimension()==3) brgType =3;

    char field_name[256];
    int num_values = msi_field_getNumVal(field);
    int total_num_dof = msi_field_getSize(field);

    int dofPerVar=total_num_dof/num_values;
 
    int num_vtx = pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);
    PetscInt firstRow, lastRowPlusOne;
    ierr = MatGetOwnershipRange(*A, &firstRow, &lastRowPlusOne);

    std::map<int, std::vector<int> > idxSendBuff, idxRecvBuff;
    std::map<int, std::vector<PetscScalar> > valuesSendBuff, valuesRecvBuff;
    int blockMatSize = total_num_dof*total_num_dof;
    for (std::map<int, std::map<int, int> > ::iterator it = remoteNodeRow.begin(); it!=remoteNodeRow.end(); it++)
    {
      idxSendBuff[it->first].resize(it->second.size()+remoteNodeRowSize[it->first]);
      valuesSendBuff[it->first].resize(remoteNodeRowSize[it->first]*blockMatSize);
      int idxOffset=0;
      int valueOffset=0;
      for(std::map<int, int> ::iterator it2 =it->second.begin(); it2!=it->second.end();it2++)
      {
        idxSendBuff[it->first].at(idxOffset++)=it2->second;
        apf::MeshEntity* ent = msi_solver::instance()->vertices[it2->first];

        std::vector<apf::MeshEntity*> vecAdj;
        apf::Adjacent elements;
        getBridgeAdjacent(pumi::instance()->mesh, ent, brgType, 0, elements);
        for (int i=0; i<elements.getSize(); ++i)
        {
          if (!pumi::instance()->mesh->isGhost(elements[i]))
            vecAdj.push_back(elements[i]);
        }
        vecAdj.push_back(ent);
        int numAdj = vecAdj.size();
        assert(numAdj==it2->second);
        std::vector<int> localNodeId(numAdj);
        std::vector<int> columns(total_num_dof*numAdj);
        for(int i=0; i<numAdj; i++)
        {
          int local_id = msi_node_getID(vecAdj.at(i), 0);
          localNodeId.at(i)=local_id;
          int start_global_dof_id, end_global_dof_id_plus_one;
          msi_node_getGlobalFieldID(field, vecAdj.at(i), 0, &start_global_dof_id, &end_global_dof_id_plus_one);
          idxSendBuff[it->first].at(idxOffset++)=start_global_dof_id;
        }
        int offset=0;
        for(int i=0; i<numAdj; i++)
        {
          int startColumn = localNodeId.at(i)*total_num_dof;
          for(int j=0; j<total_num_dof; j++)
            columns.at(offset++)=startColumn+j;
        }
        ierr = MatGetValues(remoteA, total_num_dof, &columns.at(total_num_dof*(numAdj-1)), total_num_dof*numAdj, &columns[0], &valuesSendBuff[it->first].at(valueOffset));
        valueOffset+=it2->second*blockMatSize;
      }
      assert(idxOffset==idxSendBuff[it->first].size());
      assert(valueOffset==valuesSendBuff[it->first].size());
    }
    ierr = MatDestroy(&remoteA);

    //send and receive message size
    int sendTag=2020;
    MPI_Request my_request[256];
    MPI_Status my_status[256];
    int requestOffset=0;
    std::map<int, std::pair<int, int> > msgSendSize;
    std::map<int, std::pair<int, int> > msgRecvSize;
    for(std::map<int, int >::iterator it = remoteNodeRowSize.begin(); it!=remoteNodeRowSize.end(); it++)
    {
      int destPid=it->first;
      msgSendSize[destPid].first=idxSendBuff[it->first].size();
      msgSendSize[destPid].second = valuesSendBuff[it->first].size();
      MPI_Isend(&(msgSendSize[destPid]),sizeof(std::pair<int, int>),MPI_BYTE,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
    }
    assert(requestOffset<256);
    for(std::set<int>::iterator it = remotePidOwned.begin(); it!=remotePidOwned.end(); it++)
    {
      int destPid=*it;
      MPI_Irecv(&(msgRecvSize[destPid]),sizeof(std::pair<int, int>),MPI_BYTE,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
    }
    assert(requestOffset<256);
    MPI_Waitall(requestOffset,my_request,my_status);
    //set up receive buff
    for(std::map<int, std::pair<int, int> >::iterator it = msgRecvSize.begin(); it!= msgRecvSize.end(); it++)
    {
      idxRecvBuff[it->first].resize(it->second.first);
      valuesRecvBuff[it->first].resize(it->second.second); 
    }
    // now get data
    sendTag=9999;
    requestOffset=0;
    for(std::map<int, int >::iterator it = remoteNodeRowSize. begin(); it!=remoteNodeRowSize.end(); it++)
    {
      int destPid=it->first;
      MPI_Isend(&(idxSendBuff[destPid].at(0)),idxSendBuff[destPid].size(),MPI_INT,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
      MPI_Isend(&(valuesSendBuff[destPid].at(0)),sizeof(PetscScalar)*valuesSendBuff[destPid].size(),MPI_BYTE,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
    }
    assert(requestOffset<256);
    for(std::set<int>::iterator it = remotePidOwned.begin(); it!=remotePidOwned.end(); it++)
    {
      int destPid=*it;
      MPI_Irecv(&(idxRecvBuff[destPid].at(0)),idxRecvBuff[destPid].size(),MPI_INT,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
      MPI_Irecv(&(valuesRecvBuff[destPid].at(0)),sizeof(PetscScalar)*valuesRecvBuff[destPid].size(),MPI_BYTE,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
    }
    assert(requestOffset<256);
    MPI_Waitall(requestOffset,my_request,my_status);

    for( std::map<int, std::vector<int> >::iterator it =idxSendBuff.begin(); it!=idxSendBuff.end(); it++)
      std::vector<int>().swap(it->second);
    for( std::map<int, std::vector<PetscScalar> >::iterator it =valuesSendBuff.begin(); it!=valuesSendBuff.end(); it++)
      std::vector<PetscScalar>().swap(it->second);
    valuesSendBuff.clear();
    idxSendBuff.clear();

    // now assemble the matrix
    for(std::set<int>::iterator it = remotePidOwned.begin(); it!=remotePidOwned.end(); it++)
    {
      int destPid=*it;
      int valueOffset=0;
      int idxOffset=0;
      vector<int> & idx = idxRecvBuff[destPid];
      vector<PetscScalar> & values = valuesRecvBuff[destPid];
      int numValues=values.size();
      while(valueOffset<numValues)
      {
        int numAdj = idx.at(idxOffset++); 
        std::vector<int> columns(total_num_dof*numAdj);
        int offset=0;
        for(int i=0; i<numAdj; i++, idxOffset++)
        {
          for(int j=0; j<total_num_dof; j++)
          {
            columns.at(offset++)=idx.at(idxOffset)+j;
          }
        }
        assert (columns.at(total_num_dof*(numAdj-1))>=firstRow && *columns.rbegin()<lastRowPlusOne);
        ierr = MatSetValues(*A, total_num_dof, &columns.at(total_num_dof*(numAdj-1)), total_num_dof*numAdj, &columns[0], &values.at(valueOffset),ADD_VALUES);
        valueOffset+=blockMatSize*numAdj;
      }
      std::vector<int>().swap(idxRecvBuff[destPid]);
      std::vector<PetscScalar>().swap(valuesRecvBuff[destPid]);
    }
    valuesRecvBuff.clear();
    idxRecvBuff.clear();

  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY); 
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  mat_status=MSI_FIXED;
}

int msi_matrix::flushAssembly()
{
  PetscErrorCode ierr;
  ierr = MatAssemblyBegin(*A, MAT_FLUSH_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FLUSH_ASSEMBLY);
  CHKERRQ(ierr);
}
int matrix_solve:: set_bc( int row)
{
#ifdef DEBUG
  PetscInt firstRow, lastRowPlusOne;
  int ierr = MatGetOwnershipRange(*A, &firstRow, &lastRowPlusOne);
  assert (row>=firstRow && row<lastRowPlusOne);
#endif
  MatSetValue(*A, row, row, 1.0, ADD_VALUES);
}

int matrix_solve:: set_row( int row, int numVals, int* columns, double * vals)
{
#ifdef DEBUG
  PetscInt firstRow, lastRowPlusOne;
  int ierr = MatGetOwnershipRange(*A, &firstRow, &lastRowPlusOne);
  assert (row>=firstRow && row<lastRowPlusOne);
#endif
  for(int i=0; i<numVals; i++)
  {
#ifndef PETSC_USE_COMPLEX
    set_value(row, columns[i], 1, vals[i], 0);
#else
    set_value(row, columns[i], 1, vals[2*i], vals[2*i+1]); 
#endif
  }
}
int  msi_matrix::preAllocateParaMat()
{
  int bs=1;
  MatType type;
  MatGetType(*A, &type);

  int num_own_ent,num_own_dof=0, vertex_type=0;
  num_own_ent = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
  msi_field_getnumowndof(field, &num_own_dof);
  int dofPerEnt=0;
  if (num_own_ent) dofPerEnt = num_own_dof/num_own_ent;

  if (strcmp(type, MATSEQAIJ)==0 || strcmp(type, MATMPIAIJ)==0) 
    bs=1;
  else 
    bs=dofPerEnt;
  int numBlocks = num_own_dof / bs;
  int numBlockNode = dofPerEnt / bs;
  std::vector<PetscInt> dnnz(numBlocks), onnz(numBlocks);
  int startDof, endDofPlusOne;
  msi_field_getowndofid (field, &startDof, &endDofPlusOne);

  int num_vtx=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  int nnzStash=0;
  int brgType = pumi::instance()->mesh->getDimension();
  int start_global_dof_id, end_global_dof_id_plus_one;

  apf::MeshEntity* ent;
  pMeshIter it = pumi::instance()->mesh->begin(0);  
  while ((ent = pumi::instance()->mesh->iterate(it)))
  {
    msi_node_getGlobalFieldID(field, ent, 0, &start_global_dof_id, &end_global_dof_id_plus_one);
    int startIdx = start_global_dof_id;
    if(start_global_dof_id<startDof || start_global_dof_id>=endDofPlusOne)
    {
      apf::Adjacent elements;
      getBridgeAdjacent(pumi::instance()->mesh, ent, brgType, 0, elements);
      int num_elem=0;
      for (int i=0; i<elements.getSize(); ++i)
      {
        if (!pumi::instance()->mesh->isGhost(elements[i]))
          ++num_elem;
      }

      nnzStash+=dofPerEnt*dofPerEnt*(num_elem+1);
      continue;
    }
    startIdx -= startDof;
    startIdx /=bs; 

    int adjNodeOwned, adjNodeGlb;
    pumi::instance()->mesh->getIntTag(ent, msi_solver::instance()->num_global_adj_node_tag, &adjNodeGlb);
    pumi::instance()->mesh->getIntTag(ent, msi_solver::instance()->num_own_adj_node_tag, &adjNodeOwned);
    assert(adjNodeGlb>=adjNodeOwned);

    for(int i=0; i<numBlockNode; i++)
    {
      dnnz.at(startIdx+i)=(1+adjNodeOwned)*numBlockNode;
      onnz.at(startIdx+i)=(adjNodeGlb-adjNodeOwned)*numBlockNode;
    }
  }
  pumi::instance()->mesh->end(it);
  if (bs==1) 
    MatMPIAIJSetPreallocation(*A, 0, &dnnz[0], 0, &onnz[0]);
  else  
    MatMPIBAIJSetPreallocation(*A, bs, 0, &dnnz[0], 0, &onnz[0]);
} 

int matrix_solve::setUpRemoteAStruct()
{
  int vertex_type=0;
  int num_values = msi_field_getNumVal(field);
  int total_num_dof = msi_field_getSize(field);

  int dofPerVar=total_num_dof/num_values;

  int num_vtx = pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  std::vector<int> nnz_remote(num_values*num_vtx);
  int brgType = 2;
  if (pumi::instance()->mesh->getDimension()==3) brgType =3;
  
  apf::MeshEntity* ent;
  pMeshIter it = pumi::instance()->mesh->begin(0);  
  int inode=0;
  while ((ent = pumi::instance()->mesh->iterate(it)))
  {
    int owner=pumi_ment_getOwnPID(ent);
    if (owner!=PCU_Comm_Self())
    {
      apf::Adjacent elements;
      getBridgeAdjacent(pumi::instance()->mesh, ent, brgType, 0, elements);
      int num_elem=0;
      for (int i=0; i<elements.getSize(); ++i)
      {
        if (!pumi::instance()->mesh->isGhost(elements[i]))
          ++num_elem;
      }

      remoteNodeRow[owner][inode]=num_elem+1;
      remoteNodeRowSize[owner]+=num_elem+1;
      for(int i=0; i<num_values; i++)
        nnz_remote[inode*num_values+i]=(num_elem+1)*num_values;
    }
    else 
    {
      apf::Copies remotes;
      pumi::instance()->mesh->getRemotes(ent,remotes);
      APF_ITERATE(apf::Copies, remotes, it)
        remotePidOwned.insert(it->first);
    }
    ++inode;
  }
  pumi::instance()->mesh->end(it);

  PetscErrorCode ierr = MatCreate(PETSC_COMM_SELF,&remoteA);
  CHKERRQ(ierr);
  ierr = MatSetType(remoteA, MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSetBlockSize(remoteA, dofPerVar); CHKERRQ(ierr);
  ierr = MatSetSizes(remoteA, total_num_dof*num_vtx, total_num_dof*num_vtx, PETSC_DECIDE, PETSC_DECIDE); CHKERRQ(ierr);
  MatSeqBAIJSetPreallocation(remoteA, dofPerVar, 0, &nnz_remote[0]);
  ierr = MatSetUp (remoteA);CHKERRQ(ierr);

}
int  msi_matrix::preAllocateSeqMat()
{
  int bs=1, vertex_type=0;
  MatType type;
  MatGetType(*A, &type);

  int num_vtx=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);
  int num_dof = num_vtx*apf::countComponents(field);
#ifdef PETSC_USE_COMPLEX
  num_dof /= 2;
#endif

  int dofPerEnt=0;
  if (num_vtx) dofPerEnt = num_dof/num_vtx;

  if (strcmp(type, MATSEQAIJ)==0 || strcmp(type, MATMPIAIJ)==0) 
    bs=1;
  else 
    bs=dofPerEnt;
  int numBlocks = num_dof / bs;
  int numBlockNode = dofPerEnt / bs;
  std::vector<PetscInt> nnz(numBlocks);
  int brgType = 2;
  if (pumi::instance()->mesh->getDimension()==3) brgType = 3;

  apf::MeshEntity* ent;
  pMeshIter it = pumi::instance()->mesh->begin(0);  
  int inode=0, start_dof, end_dof_plus_one;
  while ((ent = pumi::instance()->mesh->iterate(it)))
  {
    msi_node_getFieldID(field, ent, 0, &start_dof, &end_dof_plus_one);
    int startIdx = start_dof;
    assert(startIdx<num_dof);

    apf::Adjacent elements;
    getBridgeAdjacent(pumi::instance()->mesh, ent, brgType, 0, elements);
    int numAdj=0;
    for (int i=0; i<elements.getSize(); ++i)
    {
      if (!pumi::instance()->mesh->isGhost(elements[i]))
        ++numAdj;
    }

    startIdx /=bs; 
    for(int i=0; i<numBlockNode; i++)
    {
      nnz.at(startIdx+i)=(1+numAdj)*numBlockNode;
    }
    ++inode;
  }
  pumi::instance()->mesh->end(it);

  if (bs==1) 
    MatSeqAIJSetPreallocation(*A, 0, &nnz[0]);
  else  
    MatSeqBAIJSetPreallocation(*A, bs, 0, &nnz[0]);
} 

int msi_matrix::setupParaMat()
{
  int num_own_ent, vertex_type=0, num_own_dof;
  num_own_ent = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
  msi_field_getnumowndof(field, &num_own_dof);
  int dofPerEnt=0;
  if (num_own_ent) dofPerEnt = num_own_dof/num_own_ent;
  PetscInt mat_dim = num_own_dof;

  // create matrix
  PetscErrorCode ierr = MatCreate(MPI_COMM_WORLD, A);
  CHKERRQ(ierr);
  // set matrix size
  ierr = MatSetSizes(*A, mat_dim, mat_dim, PETSC_DECIDE, PETSC_DECIDE); CHKERRQ(ierr);

  ierr = MatSetType(*A, MATMPIAIJ);CHKERRQ(ierr);
}

int msi_matrix::setupSeqMat()
{
  int num_ent=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  int num_dof =  num_ent*apf::countComponents(field);
#ifdef PETSC_USE_COMPLEX
  num_dof /= 2;
#endif

  int dofPerEnt=0;
  if (num_ent) dofPerEnt = num_dof/num_ent;

  PetscInt mat_dim = num_dof;

  // create matrix
  PetscErrorCode ierr = MatCreate(PETSC_COMM_SELF, A);
  CHKERRQ(ierr);
  // set matrix size
  ierr = MatSetSizes(*A, mat_dim, mat_dim, PETSC_DECIDE, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
}

int matrix_solve::setupMat()
{
  setupParaMat();
}

int matrix_mult::setupMat()
{
  if (localMat) setupSeqMat();
  else setupParaMat();
}

int matrix_solve::preAllocate ()
{
  preAllocateParaMat();
}

int matrix_mult::preAllocate ()
{
  if(localMat) preAllocateSeqMat();
  else preAllocateParaMat();
}

#define FIXSIZEBUFF 1024
#define C1TRIDOFNODE 6
int copyField2PetscVec(pField f, Vec& petscVec)
{
  int scalar_type=0;
#ifdef PETSC_USE_COMPLEX
  scalar_type=1;
#endif
  int num_own_ent,num_own_dof=0, vertex_type=0;
  num_own_ent = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
  msi_field_getnumowndof(f, &num_own_dof);
  int dofPerEnt=0;
  if (num_own_ent) dofPerEnt = num_own_dof/num_own_ent;

  int ierr = VecCreateMPI(MPI_COMM_WORLD, num_own_dof, PETSC_DECIDE, &petscVec);
  CHKERRQ(ierr);
  VecAssemblyBegin(petscVec);

  int num_vtx=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  double dof_data[FIXSIZEBUFF];
  assert(sizeof(dof_data)>=dofPerEnt*2*sizeof(double));
  int nodeCounter=0, start_global_dof_id, end_global_dof_id_plus_one;

  apf::MeshEntity* ent;
  pMeshIter it = pumi::instance()->mesh->begin(0);  
  while ((ent = pumi::instance()->mesh->iterate(it)))
  {
    if (!pumi_ment_isOwned(ent)) continue;
    ++nodeCounter;
    int num_dof = msi_node_getField(f, ent, 0, dof_data);
    msi_node_getGlobalFieldID (f, ent, 0, &start_global_dof_id, &end_global_dof_id_plus_one);
    int startIdx=0;
    for(int i=0; i<dofPerEnt; i++)
    { 
      PetscScalar value;
#ifndef PETSC_USE_COMPLEX
      value = dof_data[startIdx++];
#else
      value = complex<double>(dof_data[startIdx*2],dof_data[startIdx*2+1]);
      startIdx++;
#endif
      ierr = VecSetValue(petscVec, start_global_dof_id+i, value, INSERT_VALUES);
      CHKERRQ(ierr);
    }
  }
  pumi::instance()->mesh->end(it);

  assert(nodeCounter==num_own_ent);
  ierr=VecAssemblyEnd(petscVec);
  CHKERRQ(ierr);
  return 0;
}

int copyPetscVec2Field(Vec& petscVec, pField f)
{
  int scalar_type=0;
#ifdef PETSC_USE_COMPLEX
  scalar_type=1;
#endif
  int num_own_ent,num_own_dof=0, vertex_type=0;
  num_own_ent = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
  msi_field_getnumowndof(f, &num_own_dof);
  int dofPerEnt=0;
  if (num_own_ent) dofPerEnt = num_own_dof/num_own_ent;

  std::vector<PetscInt> ix(dofPerEnt);
  std::vector<PetscScalar> values(dofPerEnt);
  std::vector<double> dof_data(dofPerEnt*(1+scalar_type));
  int num_vtx=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  int ierr;

  apf::MeshEntity* ent;
  pMeshIter it = pumi::instance()->mesh->begin(0);  
  while ((ent = pumi::instance()->mesh->iterate(it)))
  {
    if (!pumi_ment_isOwned(ent)) continue;
    int start_global_dof_id, end_global_dof_id_plus_one;
    msi_node_getGlobalFieldID(f, ent, 0, &start_global_dof_id, &end_global_dof_id_plus_one);
    int startIdx = start_global_dof_id;
    
    for(int i=0; i<dofPerEnt; i++)
      ix.at(i)=startIdx+i;
    ierr=VecGetValues(petscVec, dofPerEnt, &ix[0], &values[0]); CHKERRQ(ierr);
    startIdx=0;
    for(int i=0; i<dofPerEnt; i++)
    {
#ifndef PETSC_USE_COMPLEX
        dof_data.at(startIdx++)= values.at(i);
#else
        dof_data.at(2*startIdx)=values.at(i).real();
        dof_data.at(2*startIdx+1)=values.at(i).imag();
        startIdx++;
#endif
    }
    msi_node_setField(f, ent, 0, dofPerEnt, &dof_data[0]);
  }
  pumi::instance()->mesh->end(it);
  pumi_field_synchronize(f);
  return 0;
}

int matrix_solve::solve(pField rhs, pField sol)
{
  Vec x, b;
  copyField2PetscVec(rhs, b);
  int ierr = VecDuplicate(b, &x);CHKERRQ(ierr);
  //std::cout<<" before solve "<<std::endl;
  //VecView(b, PETSC_VIEWER_STDOUT_WORLD);
  if(!kspSet) setKspType();
  ierr = KSPSolve(*ksp, b, x);
  CHKERRQ(ierr);
  PetscInt its;
  ierr = KSPGetIterationNumber(*ksp, &its);
  CHKERRQ(ierr);
  int iter_num=its;
  if (PCU_Comm_Self() == 0)
    std::cout <<"\t-- # solver iterations " << its << std::endl;
  iterNum = its;
  //VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  copyPetscVec2Field(x, sol);
  ierr = VecDestroy(&b); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
}

int matrix_solve:: setKspType()
{
  PetscErrorCode ierr;
  ierr = KSPCreate(MPI_COMM_WORLD, ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(*ksp, *A, *A /*, SAME_PRECONDITIONER DIFFERENT_NONZERO_PATTERN*/);CHKERRQ(ierr);
  ierr = KSPSetTolerances(*ksp, .000001, .000000001,
                          PETSC_DEFAULT, 1000);CHKERRQ(ierr);
  int num_values = msi_field_getNumVal(field);
  int total_num_dof = msi_field_getSize(field);
  char field_name[FIXSIZEBUFF];
  strcpy(field_name, apf::getName(field));

  assert(total_num_dof/num_values==C1TRIDOFNODE*(pumi::instance()->mesh->getDimension()-1));
  // if 2D problem use superlu
  if (pumi::instance()->mesh->getDimension()==2)
  {
#ifdef PETSC_USE_COMPLEX 
    ierr=KSPSetType(*ksp, KSPPREONLY);CHKERRQ(ierr);
    PC pc;
    ierr=KSPGetPC(*ksp, &pc); CHKERRQ(ierr);
    ierr=PCSetType(pc,PCLU); CHKERRQ(ierr);
    ierr=PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST);  CHKERRQ(ierr);
#else
    if(1||num_values==1)
    {
      ierr=KSPSetType(*ksp, KSPPREONLY);CHKERRQ(ierr);
      PC pc;
      ierr=KSPGetPC(*ksp, &pc); CHKERRQ(ierr);
      ierr=PCSetType(pc,PCLU); CHKERRQ(ierr);
      ierr=PCFactorSetMatSolverPackage(pc, MATSOLVERSUPERLU_DIST);  CHKERRQ(ierr);
    }
    else
    {
      ierr=KSPSetType(*ksp, KSPFGMRES);CHKERRQ(ierr);
      //int n;
      //double rnorm;
      //KSPMonitorDefault(*ksp,n, rnorm, NULL);
      PC pc;
      PC *subpc;
      ierr=KSPGetPC(*ksp, &pc); CHKERRQ(ierr);
      ierr=PCSetType(pc,PCFIELDSPLIT); CHKERRQ(ierr);
      ierr =  PCFieldSplitSetBlockSize(pc,total_num_dof/num_values); CHKERRQ(ierr);
      for(int i=0; i<num_values; i++)
      {
        sprintf(field_name, "%dth",i);
        ierr =  PCFieldSplitSetFields(pc, field_name, 1, &i, &i);
      }
      if(num_values==2) PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR);
      ierr =  KSPSetUp(*ksp); CHKERRQ(ierr);
      KSP * subksp;
      int numSplit=-1;
      ierr = PCFieldSplitGetSubKSP(pc, &numSplit, &subksp);  CHKERRQ(ierr);
      assert(numSplit==num_values);
      for(int i=0; i<numSplit; i++)
      {
        ierr=KSPSetType(subksp[i], KSPPREONLY);CHKERRQ(ierr);
        PC pc;
        ierr=KSPGetPC(subksp[i], &pc); CHKERRQ(ierr);
        ierr=PCSetType(pc,PCLU); CHKERRQ(ierr); 
        ierr=PCFactorSetMatSolverPackage(pc, MATSOLVERSUPERLU_DIST);  CHKERRQ(ierr);
      }
      PetscFree(subksp);
    }
#endif
  }
  ierr = KSPSetFromOptions(*ksp);CHKERRQ(ierr);
  kspSet=1;
}

int msi_matrix::write (const char* file_name)
{
  PetscErrorCode ierr;
  PetscViewer lab;
  if(get_type()==0)
  {
    char name_buff[256];
    sprintf(name_buff, "%s-%d.m",file_name,PCU_Comm_Self());
    ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, name_buff, &lab); CHKERRQ(ierr);
  }
  else
  {
    ierr = PetscViewerASCIIOpen(MPI_COMM_WORLD, file_name, &lab); CHKERRQ(ierr);
  }
  ierr = PetscViewerPushFormat(lab, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
  ierr = MatView(*A, lab); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&lab); CHKERRQ(ierr);
}
int msi_matrix::printInfo()
{
  MatInfo info;
  MatGetInfo(*A, MAT_LOCAL,&info);
  std::cout<<"\t nz_allocated,nz_used,nz_unneeded "<<info.nz_allocated<<" "<<info.nz_used<<" "<<info.nz_unneeded<<std::endl;
  std::cout<<"\t memory mallocs "<<info.memory<<" "<<info.mallocs<<std::endl; 
  PetscInt nstash, reallocs, bnstash, breallocs;
  MatStashGetInfo(*A,&nstash,&reallocs,&bnstash,&breallocs);
  std::cout<<"\t nstash, reallocs, bnstash, breallocs "<<nstash<<" "<<reallocs<<" "<<bnstash<<" "<<breallocs<<std::endl;
}
#endif //#ifndef MSI_MESHGEN
