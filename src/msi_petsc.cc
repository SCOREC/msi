/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#ifdef MSI_PETSC
#include "msi.h"
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

#ifdef MSI_COMPLEX
#include "petscsys.h" // for PetscComplex
#include <complex>
using std::complex;
#endif

#define FIXSIZEBUFF 1024

using std::vector;
using std::set;

int copyField2PetscVec(pField field, Vec& petscVec, int scalar_type);
int copyPetscVec2Field(Vec& petscVec, pField field, int scalar_type);

void printMemStat()
{
  PetscLogDouble mem, mem_max;
  PetscMemoryGetCurrentUsage(&mem);
  PetscMemoryGetMaximumUsage(&mem_max);
  std::cout<<"\tMemory usage (MB) reported by PetscMemoryGetCurrentUsage: Rank "<<PCU_Comm_Self()<<" current "<<mem/1e6<<std::endl;
}
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

msi_solver::~msi_solver()
{
  if (matrix_container!=NULL)
    matrix_container->clear();
  matrix_container=NULL;
  delete _instance;
}

// ***********************************
// 		MSI_MATRIX
// ***********************************

void msi_solver::add_matrix(int matrix_id, msi_matrix* matrix)
{
  assert(matrix_container->find(matrix_id)==matrix_container->end());
  matrix_container->insert(std::map<int, msi_matrix*>::value_type(matrix_id, matrix));
}

msi_matrix* msi_solver::get_matrix(int matrix_id)
{
  std::map<int, msi_matrix*>::iterator mit = matrix_container->find(matrix_id);
  if (mit == matrix_container->end()) 
    return (msi_matrix*)NULL;
  return mit->second;
}


int matrix_solve::initialize()
{
  // initialize matrix
  setupMat();
  preAllocate();
  if(!msi_solver::instance()->assembleOption) setUpRemoteAStruct();
  int ierr = MatSetUp (*A); // "MatSetUp" sets up internal matrix data structure for the later use
  //disable error when preallocate not enough
  //check later
  ierr = MatSetOption(*A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE); 
  CHKERRQ(ierr);
  //ierr = MatSetOption(*A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE); CHKERRQ(ierr);
  ierr = MatSetOption(*A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE); 
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
  //ierr = MatSetOption(*A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE); CHKERRQ(ierr);
  ierr = MatSetOption(*A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE); CHKERRQ(ierr);
  CHKERRQ(ierr);
}

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

msi_matrix::msi_matrix(int i, pField f): id(i), field(f)
{
// initialize tag
  num_global_adj_node_tag = pumi::instance()->mesh->createIntTag("m3dc1_num_global_adj_node", 1);
  num_own_adj_node_tag = pumi::instance()->mesh->createIntTag("m3dc1_num_own_adj_node", 1);
  set_adj_node_tag(num_global_adj_node_tag, num_own_adj_node_tag);

#ifdef MSI_COMPLEX
  scalar_type=1;
#else
  scalar_type=0;
#endif
  mat_status = MSI_NOT_FIXED;
  A=new Mat;
}

int msi_matrix::destroy()
{
  pMesh mesh = pumi::instance()->mesh;
  apf::removeTagFromDimension(mesh, num_global_adj_node_tag, 0);
  mesh->destroyTag(num_global_adj_node_tag);
  apf::removeTagFromDimension(mesh, num_own_adj_node_tag, 0);
  mesh->destroyTag(num_own_adj_node_tag);

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
  assert(mat_status != MSI_FIXED);

  PetscErrorCode ierr;
  
#ifndef MSI_COMPLEX
  if (operation)
    ierr = MatSetValue(*A, row, col, real_val, ADD_VALUES);
  else
    ierr = MatSetValue(*A, row, col, real_val, INSERT_VALUES);
#else
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
  assert (mat_status != MSI_FIXED);

  PetscErrorCode ierr;
  vector<PetscScalar> petscValues(rsize*csize);
  for(int i=0; i<rsize; ++i)
  {
    for(int j=0; j<csize; ++j)
    {
#ifndef MSI_COMPLEX
      petscValues.at(i*csize+j)=values[i*csize+j];
      ierr = MatSetValues(*A, rsize, rows, csize, columns, &petscValues[0], ADD_VALUES);
#else
      petscValues.at(i*csize+j)=complex<double>(values[2*i*csize+2*j], values[2*i*csize+2*j+1]);
      ierr = MatSetValues(*A, rsize, rows, csize, columns, (PetscScalar*)values, ADD_VALUES);
#endif
    }
  }
  CHKERRQ(ierr);
}

void matrix_solve::add_blockvalues(int rbsize, int * rows, int cbsize, int * columns, double* values)
{
  int bs;
  MatGetBlockSize(remoteA, &bs);
  vector<PetscScalar> petscValues(rbsize*cbsize*bs*bs);
  //std::cout<<PCU_Comm_Self()<<" bs "<<bs<<std::endl;
  //for(int i=0; i<rbsize; ++i) std::cout<<" row "<<rows[i]<<" "<<std::endl;
  //for(int i=0; i<cbsize; ++i) std::cout<<" columns "<<columns[i]<<" "<<std::endl;
  for(int i=0; i<rbsize*bs; ++i)
  {
    for(int j=0; j<cbsize*bs; ++j)
    {
#ifndef MSI_COMLEX
      petscValues.at(i*cbsize*bs+j)=values[i*cbsize*bs+j];
      MatSetValuesBlocked(remoteA,rbsize, rows, cbsize, columns, &petscValues[0], ADD_VALUES);
#else
      petscValues.at(i*cbsize*bs+j)=complex<double>(values[2*i*cbsize*bs+2*j], values[2*i*cbsize*bs+2*j+1]);
      MatSetValuesBlocked(remoteA,rbsize, rows, cbsize, columns, (PetscScalar*)values, ADD_VALUES);
#endif
    }
  }
}

int msi_matrix::get_values(vector<int>& rows, vector<int>& n_columns, vector<int>& columns, vector<double>& values)
{
  assert (mat_status == MSI_FIXED);

#ifdef MSI_COMPLEX
   if (!PCU_Comm_Self())
     std::cout<<"[MSI ERROR] "<<__func__<<": not supported for complex\n";
   return;
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
#endif
}

// ***********************************
// 		matrix_mult
// ***********************************

int matrix_mult::multiply(pField in_field, pField out_field)
{
  if (!localMat)
  {
    Vec b, c;
    copyField2PetscVec(in_field, b, get_scalar_type());
    int ierr = VecDuplicate(b, &c);
    CHKERRQ(ierr);

    MatMult(*A, b, c);

    copyPetscVec2Field(c, out_field, get_scalar_type());
    ierr = VecDestroy(&b); 
    CHKERRQ(ierr);
    ierr = VecDestroy(&c); 
    CHKERRQ(ierr);
  }
  else
  {
    Vec b, c;
   
    int num_dof = apf::countComponents(in_field);
#ifdef MSI_COMPLEX
    num_dof/=2;
#endif
    num_dof *= pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

    int bs;
    int ierr;
    MatGetBlockSize(*A, &bs);
    PetscScalar *array[2];
    array[0] = (PetscScalar*) apf::getArrayData(in_field);
    ierr = VecCreateSeqWithArray( PETSC_COMM_SELF, bs, num_dof, (PetscScalar*) array[0],&b); CHKERRQ(ierr);

    array[1] = (PetscScalar*) apf::getArrayData(out_field);
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
matrix_solve::matrix_solve(int i, pField f): msi_matrix(i,f) 
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
  if(mat_status==MSI_NOT_FIXED && msi_solver::instance()->assembleOption==0) MatDestroy(&remoteA);
}

int matrix_solve::assemble()
{
  PetscErrorCode ierr;
  double t1 = MPI_Wtime(), t2=t1;
 // if (!PCU_Comm_Self())
 // {
 //   std::cout<<"Before assemble"<<std::endl;
 //   printMemStat();
  //}
  if(!msi_solver::instance()->assembleOption)
  {
    ierr = MatAssemblyBegin(remoteA, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(remoteA, MAT_FINAL_ASSEMBLY);
    t2 = MPI_Wtime();
    //if (!PCU_Comm_Self()) std::cout<<"\t Assembly remoteA time "<<t2-t1<<std::endl;
    //ierr = MatView(remoteA, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
    //pass remoteA to ownnering process
    int brgType = 2;
    if (pumi::instance()->mesh->getDimension()==3) brgType =3;

   int total_num_dof = apf::countComponents(field);
#ifdef MSI_COMPLEX
    total_num_dof/=2;
#endif
    int dofPerVar=total_num_dof/num_values;
 
    int num_vtx = pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);
    PetscInt firstRow, lastRowPlusOne;
    ierr = MatGetOwnershipRange(*A, &firstRow, &lastRowPlusOne);

    std::map<int, std::vector<int> > idxSendBuff, idxRecvBuff;
    std::map<int, std::vector<PetscScalar> > valuesSendBuff, valuesRecvBuff;
    int blockMatSize = total_num_dof*total_num_dof;
    for (std::map<int, std::map<int, int> > ::iterator it = remoteNodeRow.begin(); it!=remoteNodeRow.end(); ++it)
    {
      idxSendBuff[it->first].resize(it->second.size()+remoteNodeRowSize[it->first]);
      valuesSendBuff[it->first].resize(remoteNodeRowSize[it->first]*blockMatSize);
      int idxOffset=0;
      int valueOffset=0;
      for(std::map<int, int> ::iterator it2 =it->second.begin(); it2!=it->second.end();++it2)
      {
        idxSendBuff[it->first].at(idxOffset++)=it2->second;
        apf::MeshEntity* ent = pumi_mesh_findEnt(pumi::instance()->mesh, 0, it2->first);

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
        for(int i=0; i<numAdj; ++i)
        {
          int local_id = pumi_ment_getID(vecAdj.at(i));
          localNodeId.at(i)=local_id;
          int start_global_dof_id, end_global_dof_id_plus_one;
          msi_ment_getGlobalFieldID (vecAdj[i], field, &start_global_dof_id, &end_global_dof_id_plus_one);
          idxSendBuff[it->first].at(idxOffset++)=start_global_dof_id;
        }
        int offset=0;
        for(int i=0; i<numAdj; ++i)
        {
          int startColumn = localNodeId.at(i)*total_num_dof;
          for(int j=0; j<total_num_dof; ++j)
            columns.at(offset++)=startColumn+j;
        }
        ierr = MatGetValues(remoteA, total_num_dof, &columns.at(total_num_dof*(numAdj-1)), total_num_dof*numAdj, &columns[0], &valuesSendBuff[it->first].at(valueOffset));
        //for(int i=0; i<total_num_dof*numAdj; ++i)
          //std::cout<<" get values indx "<<columns.at(i)<<std::endl;
        //for(int i=0; i<it2->second*blockMatSize; ++i)
          //std::cout<<"values "<<i<<" "<<valuesSendBuff[it->first].at(valueOffset+i)<<std::endl;
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
    for(std::map<int, int > :: iterator it = remoteNodeRowSize.begin(); it!=remoteNodeRowSize.end(); ++it)
    {
      int destPid=it->first;
      msgSendSize[destPid].first=idxSendBuff[it->first].size();
      msgSendSize[destPid].second = valuesSendBuff[it->first].size();
      MPI_Isend(&(msgSendSize[destPid]),sizeof(std::pair<int, int>),MPI_BYTE,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
    }
    assert(requestOffset<256);
    for(std::set<int> :: iterator it = remotePidOwned.begin(); it!=remotePidOwned.end(); ++it)
    {
      int destPid=*it;
      MPI_Irecv(&(msgRecvSize[destPid]),sizeof(std::pair<int, int>),MPI_BYTE,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
    }
    assert(requestOffset<256);
    MPI_Waitall(requestOffset,my_request,my_status);
    //set up receive buff
    for(std::map<int, std::pair<int, int> > :: iterator it = msgRecvSize.begin(); it!= msgRecvSize.end(); ++it)
    {
      idxRecvBuff[it->first].resize(it->second.first);
      valuesRecvBuff[it->first].resize(it->second.second); 
    }
    // now get data
    sendTag=9999;
    requestOffset=0;
    for(std::map<int, int > :: iterator it = remoteNodeRowSize. begin(); it!=remoteNodeRowSize.end(); ++it)
    {
      int destPid=it->first;
      MPI_Isend(&(idxSendBuff[destPid].at(0)),idxSendBuff[destPid].size(),MPI_INT,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
      MPI_Isend(&(valuesSendBuff[destPid].at(0)),sizeof(PetscScalar)*valuesSendBuff[destPid].size(),MPI_BYTE,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
    }
    assert(requestOffset<256);
    for(std::set<int> :: iterator it = remotePidOwned.begin(); it!=remotePidOwned.end(); ++it)
    {
      int destPid=*it;
      MPI_Irecv(&(idxRecvBuff[destPid].at(0)),idxRecvBuff[destPid].size(),MPI_INT,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
      MPI_Irecv(&(valuesRecvBuff[destPid].at(0)),sizeof(PetscScalar)*valuesRecvBuff[destPid].size(),MPI_BYTE,destPid,sendTag,MPI_COMM_WORLD,&(my_request[requestOffset++]));
    }
    assert(requestOffset<256);
    MPI_Waitall(requestOffset,my_request,my_status);
    //if (!PCU_Comm_Self())
    //{
    //  std::cout<<"Peak in scorec assemble"<<std::endl;
    //  printMemStat();
    //}

    for( std::map<int, std::vector<int> > :: iterator it =idxSendBuff.begin(); it!=idxSendBuff.end(); ++it)
      std::vector<int>().swap(it->second);
    for( std::map<int, std::vector<PetscScalar> > :: iterator it =valuesSendBuff.begin(); it!=valuesSendBuff.end(); ++it)
      std::vector<PetscScalar>().swap(it->second);
    valuesSendBuff.clear();
    idxSendBuff.clear();

    // now assemble the matrix
    for(std::set<int> :: iterator it = remotePidOwned.begin(); it!=remotePidOwned.end(); ++it)
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
        for(int i=0; i<numAdj; ++i, ++idxOffset)
        {
          for(int j=0; j<total_num_dof; ++j)
          {
            columns.at(offset++)=idx.at(idxOffset)+j;
          }
        }
        assert (columns.at(total_num_dof*(numAdj-1))>=firstRow && *columns.rbegin()<lastRowPlusOne);
        ierr = MatSetValues(*A, total_num_dof, &columns.at(total_num_dof*(numAdj-1)), total_num_dof*numAdj, &columns[0], &values.at(valueOffset),ADD_VALUES);
        /*int start_row=total_num_dof*(numAdj-1);
        for(int i=0; i<total_num_dof; ++i)
        {
          int row = columns.at(start_row++);
          for(int j=0; j<total_num_dof*numAdj; j++)
             MatSetValue(*A, row, columns.at(j), values.at(valueOffset++),ADD_VALUES);
        }*/
        valueOffset+=blockMatSize*numAdj;
      }
      std::vector<int>().swap(idxRecvBuff[destPid]);
      std::vector<PetscScalar>().swap(valuesRecvBuff[destPid]);
    }
    valuesRecvBuff.clear();
    idxRecvBuff.clear();
  }

  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY); 
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  mat_status=MSI_FIXED;
}

int msi_matrix :: flushAssembly()
{
  PetscErrorCode ierr;
  ierr = MatAssemblyBegin(*A, MAT_FLUSH_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FLUSH_ASSEMBLY);
  CHKERRQ(ierr);
}

void matrix_solve:: set_bc(int row)
{
#ifdef DEBUG
  PetscInt firstRow, lastRowPlusOne;
  int ierr = MatGetOwnershipRange(*A, &firstRow, &lastRowPlusOne);
  assert (row>=firstRow && row<lastRowPlusOne);
#endif
  MatSetValue(*A, row, row, 1.0, ADD_VALUES);
}

void matrix_solve:: set_row( int row, int numVals, int* columns, double * vals)
{
#ifdef DEBUG
  PetscInt firstRow, lastRowPlusOne;
  int ierr = MatGetOwnershipRange(*A, &firstRow, &lastRowPlusOne);
  assert (row>=firstRow && row<lastRowPlusOne);
#endif
  for(int i=0; i<numVals; ++i)
  {
#ifndef MSI_COMPLEX
    set_value(row, columns[i], 1, vals[i], 0);
#else
    set_value(row, columns[i], 1, vals[2*i], vals[2*i+1]); 
#endif
  }
}

int msi_matrix::preAllocateParaMat()
{
  int bs=1;
  MatType type;
  MatGetType(*A, &type);

  int num_own_ent = pumi_mesh_getNumOwnEnt (pumi::instance()->mesh, 0);
  int num_own_dof = msi_field_getNumOwnDOF(field);

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
  msi_field_getOwnDOFID (field, &startDof, &endDofPlusOne);

  int num_vtx=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  int nnzStash=0;
  int brgType = 2;
  if (pumi::instance()->mesh->getDimension()==3) brgType =3;

  apf::MeshEntity* ent;
  for(int inode=0; inode<num_vtx; ++inode)
  {
    ent = pumi_mesh_findEnt(pumi::instance()->mesh, 0, inode);
    int start_global_dof_id, end_global_dof_id_plus_one;
    msi_ment_getGlobalFieldID (ent, field, &start_global_dof_id, &end_global_dof_id_plus_one);
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
    pumi::instance()->mesh->getIntTag(ent, num_global_adj_node_tag, &adjNodeGlb);
    pumi::instance()->mesh->getIntTag(ent, num_own_adj_node_tag, &adjNodeOwned);
    assert(adjNodeGlb>=adjNodeOwned);

    for(int i=0; i<numBlockNode; ++i)
    {
      dnnz.at(startIdx+i)=(1+adjNodeOwned)*numBlockNode;
      onnz.at(startIdx+i)=(adjNodeGlb-adjNodeOwned)*numBlockNode;
    }
  }
  if (bs==1) 
    MatMPIAIJSetPreallocation(*A, 0, &dnnz[0], 0, &onnz[0]);
  else  
    MatMPIBAIJSetPreallocation(*A, bs, 0, &dnnz[0], 0, &onnz[0]);
  //std::cout<<" nnzStash "<<nnzStash*2<<std::endl;
  //MatStashSetInitialSize(*A, nnzStash*2, 0);
} 

int matrix_solve :: setUpRemoteAStruct()
{
  int total_num_dof = apf::countComponents(field);
#ifdef MSI_COMPLEX
  total_num_dof/=2;
#endif
  int dofPerVar=total_num_dof/num_values;

  int num_vtx = pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  std::vector<int> nnz_remote(num_values*num_vtx);
  int brgType = 2;
  if (pumi::instance()->mesh->getDimension()==3) brgType =3;
  
  apf::MeshEntity* ent;
  for(int inode=0; inode<num_vtx; ++inode)
  {
    ent = pumi_mesh_findEnt(pumi::instance()->mesh, 0, inode);
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
      for(int i=0; i<num_values; ++i)
        nnz_remote[inode*num_values+i]=(num_elem+1)*num_values;
    }
    else 
    {
      apf::Copies remotes;
      pumi::instance()->mesh->getRemotes(ent,remotes);
      APF_ITERATE(apf::Copies, remotes, it)
        remotePidOwned.insert(it->first);
    }
  }
  PetscErrorCode ierr = MatCreate(PETSC_COMM_SELF,&remoteA);
  CHKERRQ(ierr);
  ierr = MatSetType(remoteA, MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSetBlockSize(remoteA, dofPerVar); CHKERRQ(ierr);
  ierr = MatSetSizes(remoteA, total_num_dof*num_vtx, total_num_dof*num_vtx, PETSC_DECIDE, PETSC_DECIDE); CHKERRQ(ierr);
  MatSeqBAIJSetPreallocation(remoteA, dofPerVar, 0, &nnz_remote[0]);
  ierr = MatSetUp (remoteA);CHKERRQ(ierr);
}

int msi_matrix :: preAllocateSeqMat()
{
  int bs=1, vertex_type=0;
  MatType type;
  MatGetType(*A, &type);

  int num_vtx=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);
//  num_components = (*num_values)*(*scalar_type+1)*(*num_dofs_per_value);
  int num_dof = apf::countComponents(field);
#ifdef MSI_COMPLEX
  num_dof/=2;
#endif

  num_dof *= num_vtx;
  int dofPerEnt=num_dof/num_vtx;

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
  for(int inode=0; inode<num_vtx; ++inode)
  {
    ent = pumi_mesh_findEnt(pumi::instance()->mesh, 0, inode);
    int start_dof, end_dof_plus_one;
    msi_ment_getLocalFieldID (ent, field, &start_dof, &end_dof_plus_one);
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
    for(int i=0; i<numBlockNode; ++i)
    {
      nnz.at(startIdx+i)=(1+numAdj)*numBlockNode;
    }
  }
  if (bs==1) 
    MatSeqAIJSetPreallocation(*A, 0, &nnz[0]);
  else  
    MatSeqBAIJSetPreallocation(*A, bs, 0, &nnz[0]);
} 

int msi_matrix :: setupParaMat()
{
  int num_own_ent = pumi_mesh_getNumOwnEnt (pumi::instance()->mesh, 0);
  int num_own_dof = msi_field_getNumOwnDOF(field);

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

int msi_matrix :: setupSeqMat()
{
  int num_ent=pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);
  int num_dof = apf::countComponents(field);
#ifdef MSI_COMPLEX
  num_dof/=2;
#endif
  num_dof *= num_ent;

  int dofPerEnt=num_dof/num_ent;

  PetscInt mat_dim = num_dof;

  // create matrix
  PetscErrorCode ierr = MatCreate(PETSC_COMM_SELF, A);
  CHKERRQ(ierr);
  // set matrix size
  ierr = MatSetSizes(*A, mat_dim, mat_dim, PETSC_DECIDE, PETSC_DECIDE); CHKERRQ(ierr);

  //use block mpi iij as default
  //ierr = MatSetType(*A, MATSEQBAIJ);CHKERRQ(ierr);
  //ierr = MatSetBlockSize(*A, dofPerEnt); CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
}

int matrix_solve :: setupMat()
{
  setupParaMat();
}

int matrix_mult :: setupMat()
{
  if (localMat) setupSeqMat();
  else setupParaMat();
}

void matrix_solve :: preAllocate ()
{
  preAllocateParaMat();
}

void matrix_mult :: preAllocate ()
{
  if(localMat) preAllocateSeqMat();
  else preAllocateParaMat();
}

int copyField2PetscVec(pField f, Vec& petscVec, int scalar_type)
{
  int num_own_ent = pumi_mesh_getNumOwnEnt (pumi::instance()->mesh, 0);
  int num_own_dof = msi_field_getNumOwnDOF(f);

  int dofPerEnt=0;
  if (num_own_ent) dofPerEnt = num_own_dof/num_own_ent;

  int ierr = VecCreateMPI(MPI_COMM_WORLD, num_own_dof, PETSC_DECIDE, &petscVec);
  CHKERRQ(ierr);
  VecAssemblyBegin(petscVec);

  int num_vtx=pumi_mesh_getNumEnt (pumi::instance()->mesh, 0);

  double dof_data[FIXSIZEBUFF];
  assert(sizeof(dof_data)>=dofPerEnt*2*sizeof(double));
  int nodeCounter=0;

  apf::MeshEntity* ent;
  for(int inode=0; inode<num_vtx; ++inode)
  {
    ent = pumi_mesh_findEnt(pumi::instance()->mesh, 0, inode);
    if (!pumi_ment_isOwned(ent)) continue;
      ++nodeCounter;
    int num_dof;
    pumi_ment_getField (ent, f, 0, &dof_data[0]);
    assert(num_dof*(1+scalar_type)<=sizeof(dof_data)/sizeof(double));
    int start_global_dof_id, end_global_dof_id_plus_one;
    msi_ment_getGlobalFieldID (ent, f, &start_global_dof_id, &end_global_dof_id_plus_one);
    int startIdx=0;
    for(int i=0; i<dofPerEnt; ++i)
    { 
      PetscScalar value;
#ifndef MSI_COMLEX
      value = dof_data[startIdx++];
#else
      value = complex<double>(dof_data[startIdx*2],dof_data[startIdx*2+1]);
#endif
      ++startIdx;
      ierr = VecSetValue(petscVec, start_global_dof_id+i, value, INSERT_VALUES);
      CHKERRQ(ierr);
    }
  }
  assert(nodeCounter==num_own_ent);
  ierr=VecAssemblyEnd(petscVec);
  CHKERRQ(ierr);
}

int copyPetscVec2Field(Vec& petscVec, pField f, int scalar_type)
{
  int num_own_ent,num_own_dof=0, vertex_type=0;
  num_own_ent = pumi_mesh_getNumOwnEnt(pumi::instance()->mesh, 0);
  num_own_dof = msi_field_getNumOwnDOF(f);

  int dofPerEnt=0;
  if (num_own_ent) dofPerEnt = num_own_dof/num_own_ent;

  std::vector<PetscInt> ix(dofPerEnt);
  std::vector<PetscScalar> values(dofPerEnt);
  std::vector<double> dof_data(dofPerEnt*(1+scalar_type));
  int num_vtx= pumi_mesh_getNumEnt(pumi::instance()->mesh, 0);

  int ierr;

  apf::MeshEntity* ent;
  for(int inode=0; inode<num_vtx; ++inode)
  {
    ent = pumi_mesh_findEnt(pumi::instance()->mesh, 0, inode);
    if (!pumi_ment_isOwned(ent)) continue;
    int start_global_dof_id, end_global_dof_id_plus_one;
    msi_ment_getLocalFieldID (ent, f, &start_global_dof_id, &end_global_dof_id_plus_one);
    int startIdx = start_global_dof_id;
    
    for(int i=0; i<dofPerEnt; ++i)
      ix.at(i)=startIdx+i;
    ierr=VecGetValues(petscVec, dofPerEnt, &ix[0], &values[0]); CHKERRQ(ierr);
    startIdx=0;
    for(int i=0; i<dofPerEnt; ++i)
    {
#ifndef MSI_COMPLEX
        dof_data.at(startIdx++)= values.at(i);
#else
        dof_data.at(2*startIdx)=values.at(i).real();
        dof_data.at(2*startIdx+1)=values.at(i).imag();
        ++startIdx;
#endif
    }
    pumi_ment_setField (ent, f, 0, &dof_data[0]);
  }
  pumi_field_synchronize(f);
}

int matrix_solve::solve(pField f_x, pField f_b)
{
  Vec x, b;
  copyField2PetscVec(f_b, b, get_scalar_type());
  int ierr = VecDuplicate(b, &x);CHKERRQ(ierr);

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
  copyPetscVec2Field(x, f_x, get_scalar_type());
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

  // if 2D problem use superlu
  if (pumi::instance()->mesh->getDimension()==2)
  {
    ierr=KSPSetType(*ksp, KSPPREONLY);CHKERRQ(ierr);
    PC pc;
    ierr=KSPGetPC(*ksp, &pc); CHKERRQ(ierr);
    ierr=PCSetType(pc,PCLU); CHKERRQ(ierr);
    ierr=PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST);  CHKERRQ(ierr);
  }

  ierr = KSPSetFromOptions(*ksp);CHKERRQ(ierr);
  kspSet=1;
}

int msi_matrix :: write (const char* file_name)
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

void msi_matrix :: printInfo()
{
  MatInfo info;
  MatGetInfo(*A, MAT_LOCAL,&info);
  std::cout<<"Matrix "<<id<<" info "<<std::endl;
  std::cout<<"\t nz_allocated,nz_used,nz_unneeded "<<info.nz_allocated<<" "<<info.nz_used<<" "<<info.nz_unneeded<<std::endl;
  std::cout<<"\t memory mallocs "<<info.memory<<" "<<info.mallocs<<std::endl; 
  PetscInt nstash, reallocs, bnstash, breallocs;
  MatStashGetInfo(*A,&nstash,&reallocs,&bnstash,&breallocs);
  std::cout<<"\t nstash, reallocs, bnstash, breallocs "<<nstash<<" "<<reallocs<<" "<<bnstash<<" "<<breallocs<<std::endl;
}
#endif
