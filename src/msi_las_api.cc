#include "msi.h"
#include "msi_las.h"
#include <las.h>
#include <lasConfig.h>
#include <map>
std::map<Matrix*,pField> mat2fld;
void msi_las_setComm(MPI_Comm cm)
{
  LAS_COMM_WORLD = cm;
}
Matrix * msi_las_createMatrix(pField f)
{
  las::Sparsity * nnz = las::buildNNZSparsity(f);
  unsigned lcl = 0; // countLocalDofs(f);
  unsigned gbl = 0;
  unsigned bs = 0;
  Matrix * m  = (Matrix*)las::createPetscMatrix(lcl,gbl,bs,nnz);
  mat2fld[m] = f;
  return m;
}
Vector * msi_las_createVector(pField f)
{
  unsigned lcl = 0; // countLocalDofs(f);
  unsigned gbl = 0; // countGlobalDofs(f); | PCU_Comm_Sum(lcl);
  unsigned bs = 0;
  return (Vector*)las::createPetscVector(lcl,gbl,bs);
}
Vector * msi_las_createRHSVector(Matrix * m)
{
  return (Vector*)las::createRHSVector((las::Mat*)m);
}
Vector * msi_las_createLHSVector(Matrix * m);
{
  return (Vector*)las::createLHSVector((las::Mat*)m);
}
void msi_las_deleteMatrix(Matrix * m)
{
  las::destroyPetscMatrix((las::Mat*)m);
}
void msi_las_deleteVector(Vector * v)
{
  las::destroyPetscVector((las::Vec*)v);
}
void msi_las_getMatrixField(Matrix * m)
{
  return mat2fld(m);
}
void msi_las_setMatrix(Matrix * m, sz_t cntr, idx_t * rws, sz_t cntc, idx_t * cls, scalar * vls)
{
  ops->set((las::Mat*)m,cntr,rws,cntc,cls,vls);
}
void msi_las_addMatrix(Matrix * m, sz_t cntr, idx_t * rws, sz_t cntc, idx_t * cls, scalar * vls)
{
  ops->add((las::Mat*)m,cntr,rws,cntc,cls,vls);
}
void msi_las_setVector(Vector * v, sz_t cntr, idx_t * rws, scalar * vls)
{
  ops->set((las::Vec*)v,cntr,rws,vls);
}
void msi_las_addVector(Vector * v, sz_t cntr, idx_t * rws, scalar * vls)
{
  ops->add((las::Vec*)v,cntr,rws,vls);
}
void msi_las_setMatrixBC(Matrix * m, idx_t rw)
{
  ops->zero((las::Mat*)m,ROW,rw);
  ops->set((las::Mat*)m,1,&rw,
}
