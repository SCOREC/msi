/****************************************************************************** 

  (c) 2017 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#include "msi.h"
#include "pumi.h"
#include <iostream>
#include <assert.h>
#include <parma.h>
#include "PCU.h"
#include "petscksp.h"
#include <iostream>
#include <assert.h>

using namespace std;
static char help[] = "Purpose: testing pumi-petsc interface;\n\tdo mat-vec product A*b=c; solve Ax=c; compare x and b\n";

bool AlmostEqualDoubles(double A, double B,
            double maxDiff, double maxRelDiff);

const char* modelFile = 0;
const char* meshFile = 0;

void getConfig(int argc, char** argv)
{
  if (argc<3)
  {
    if (!PCU_Comm_Self()) 
      cout<<help<<"Usage: "<<argv[0]<<" model(.dmg) distributed-mesh(.smb)\n";
    PetscFinalize();
    pumi_finalize();
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  modelFile = argv[1];
  meshFile = argv[2];
}

int main( int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  pumi_start();
  
  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  PetscLogDouble mem;
  double t1, t2, t3, t4, t5, t6;

  int scalar_type=0; // real	
#ifdef PETSC_USE_COMPLEX
  scalar_type=1; // complex
#endif

  // read input args - in-model-file in-mesh-file out-mesh-file num-in-part
  getConfig(argc,argv);

  // load model
  pGeom g = pumi_geom_load(modelFile);
  pMesh m = pumi_mesh_load(g, meshFile, pumi_size());

  printStats(m);

   // set/get field dof values
  int vertex_dim=0, mesh_dim=pumi_mesh_getDim(m);

  if(!PCU_Comm_Self()) cout<<"* Start MSI ..."<<endl;
  msi_start(m, NULL);

  int num_vertex = pumi_mesh_getNumEnt(m, 0);
  int num_own_vertex = pumi_mesh_getNumOwnEnt(m, 0); 
  int num_elem = pumi_mesh_getNumEnt(m, mesh_dim);

  int nv=1, nd=6;
  if (mesh_dim==3) nd=12;

  if(!PCU_Comm_Self()) cout<<"* creating fields - "<<nv*nd<<" DOFs per node\n";
  pField b_field = msi_field_create ("b_field", nv, nd);
  pField c_field = msi_field_create ("c_field", nv, nd);
  pField x_field = msi_field_create ("x_field", nv, nd);

  pMeshEnt e = pumi_mesh_findEnt(m, mesh_dim, 0);
  int num_nodes_elem = pumi_ment_getNumAdj(e, 0);

  int num_dofs_node = nv * nd;
  int num_dofs_element = num_dofs_node*num_nodes_elem;

  if(!PCU_Comm_Self()) cout<<"* set b field ..."<<endl;
  // fill b field
  for(int inode=0; inode<num_vertex; inode++)
  {
    double xyz[3];
    pMeshEnt e = pumi_mesh_findEnt(m, 0, inode);
    pumi_node_getCoord(e, 0, xyz);
    // 2D mesh, z component =0
    vector<double> dofs(num_dofs_node*(1+scalar_type));
    for(int i=0; i<num_dofs_node*(1+scalar_type); i++)
      dofs.at(i)=xyz[i%3];
    pumi_ment_setField(e, b_field, 0, &dofs.at(0));
  }


  // fill matrix 
  // the matrix is diagnal dominant; thus should be positive definite
  pMatrix matrix_mult = msi_matrix_create(MSI_MULTIPLY, b_field);
  pMatrix matrix_solve = msi_matrix_create(MSI_SOLVE, b_field);

  double diag_value=2.0, off_diag=1.0;
  vector<double> block(num_dofs_element*num_dofs_element*(1+scalar_type),0);
  for(int i=0; i<num_dofs_element; i++)
  {
    for(int j=0; j<num_dofs_element; j++)
    {
      double val= (i==j? diag_value: off_diag);
      if(!scalar_type) block.at(i*num_dofs_element+j)=val;
      else
      {
        block.at(2*i*num_dofs_element+2*j)=val;
        block.at(2*i*num_dofs_element+2*j+1)=off_diag;
      }
    }
  }

  if (!pumi_rank())
  for (int i=0; i<num_dofs_element; ++i)
  {
    std::cout<<"block["<<i<<"] = ";
    for (int j=0; j<num_dofs_element; ++j)
      std::cout<<block[i*num_dofs_element+j]<<" ";
    std::cout<<"\n";
  }

  t1 = MPI_Wtime();
  for(int ielm=0; ielm<num_elem; ielm++)
  {
    for(int rowVar=0; rowVar<nv; rowVar++)
    {
      for(int colVar=0; colVar<nv; colVar++)
      {
         vector<double> block_tmp = block;
         if (rowVar!=colVar)
         {
           for(int i=0; i<block_tmp.size(); i++) block_tmp.at(i)*=0.5/nv;
         }
         msi_matrix_addBlock(matrix_solve, ielm, rowVar, colVar, &block_tmp[0]);
         msi_matrix_addBlock(matrix_mult, ielm, rowVar, colVar, &block_tmp[0]);
      }
    }
  }
  t2 = MPI_Wtime();

  if(!PCU_Comm_Self()) cout<<"* assemble matrix ..."<<endl;
  msi_matrix_freeze(matrix_mult);
  msi_matrix_freeze(matrix_solve); 
  t3 = MPI_Wtime();

  if(!PCU_Comm_Self()) cout<<"* multiply Ab=c ..."<<endl;
  msi_matrix_multiply(matrix_mult, b_field, c_field); 
  t4 = MPI_Wtime();

  // let's test field operations here
  pumi_field_copy(c_field, x_field);
  double val[]={2.};
  int realtype=0;
  pumi_field_multiply(x_field, 2.0, x_field);
  pumi_field_add(x_field, c_field, x_field);

  pMeshIter it = m->begin(0);  
  while ((e = m->iterate(it)))
  {
    vector<double> dofs1(num_dofs_node*(1+scalar_type)), dofs2(num_dofs_node*(1+scalar_type));
    pumi_ment_getField(e, c_field, 0, &dofs1.at(0));
    pumi_ment_getField(e, x_field, 0, &dofs2.at(0));
    for(int i=0; i<num_dofs_node*((1+scalar_type)); ++i)
      assert(AlmostEqualDoubles(dofs2.at(i), (val[0]+1)*dofs1.at(i), 1e-6, 1e-6));
  }
  m->end(it);

  // copy c field to x field
  pumi_field_copy(c_field, x_field);

  t5 = MPI_Wtime();
  if(!PCU_Comm_Self()) cout<<"* solve Ax=c ..."<<endl;
  // solve Ax=c
  int solver_type = 0;    // PETSc direct solver
  double solver_tol = 1e-6;
  msi_matrix_solve(matrix_solve, c_field, x_field); 

  t6 = MPI_Wtime();

  // verify x=b
  if(!PCU_Comm_Self()) cout<<"* verify x==b..."<<endl;
  it = m->begin(0);  
  while ((e = m->iterate(it)))
  {
    vector<double> dofs_x(num_dofs_node*(1+scalar_type)), dofs_b(num_dofs_node*(1+scalar_type));
    pumi_ment_getField(e, b_field, 0, &dofs_b.at(0));
    pumi_ment_getField(e, x_field, 0, &dofs_x.at(0));
    for(int idof=0; idof<num_dofs_node*(1+scalar_type); ++idof)
      assert(AlmostEqualDoubles(dofs_b.at(idof),dofs_x.at(idof), 1e-3, 1e-3));
  }
  m->end(it);

  if(!PCU_Comm_Self())
    cout<<"* time: fill matrix "<<t2-t1<<" assemble "<<t3-t2<<" mult "<<t4-t3<<" solve "<<t6-t5<<endl; 

  msi_matrix_delete(matrix_mult);
  msi_matrix_delete(matrix_solve);

  pumi_mesh_verify(m, false);

  pumi_field_delete(x_field);
  pumi_field_delete(b_field);
  pumi_field_delete(c_field);

  msi_finalize(m);
  pumi_mesh_delete(m);
  pumi_finalize();
  PetscFinalize();
  MPI_Finalize();
  return 0;
}

bool AlmostEqualDoubles(double A, double B,
            double maxDiff, double maxRelDiff)
{
// http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/ 
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    double diff = fabs(A - B);
    if (diff <= maxDiff)
        return true;
 
    A = fabs(A);
    B = fabs(B);
    double largest = (B > A) ? B : A;
 
    if (diff <= largest * maxRelDiff)
        return true;
    return false;
}

