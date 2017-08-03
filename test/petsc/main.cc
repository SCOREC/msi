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
#include "PCU.h"
#include "petscksp.h"

static char help[] = "testing petsc-pumi; \n do mat-vec product A*b=c; solve Ax=c; compare x and b\n\n";

bool AlmostEqualDoubles(double A, double B,
            double maxDiff, double maxRelDiff);

const char* modelFile = 0;
const char* meshFile = 0;
const char* outFile = 0;
int num_in_part = 0;

void getConfig(int argc, char** argv)
{
  if ( argc < 4 ) {
    if ( !PCU_Comm_Self() )
      printf("Usage: %s <model> <mesh> <outMesh> <num_part_in_mesh> <do_distribution(0/1)>\n", argv[0]);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  modelFile = argv[1];
  meshFile = argv[2];
  outFile = argv[3];
  if (argc>=4)
    num_in_part = atoi(argv[4]);
}

int main( int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  pumi_start();
  pumi_printSys();

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  PetscLogDouble mem;
  if (argc<4 & !PCU_Comm_Self())
  {
    std::cout<<"Usage: ./main model mesh out-mesh num_part"<<std::endl;
    return 1;
  }

  double t1, t2, t3, t4, t5;

  int scalar_type=0; // real	
#ifdef MSI_COMPLEX
  scalar_type=1; // complex
#endif

  // read input args - in-model-file in-mesh-file out-mesh-file num-in-part
  getConfig(argc,argv);

  // load model
  pGeom g = pumi_geom_load(modelFile);
  pMesh m = pumi_mesh_load(g, meshFile, num_in_part);
  pumi_mesh_verify(m, false);

  int mesh_dim = pumi_mesh_getDim(m);
  int num_vertex = pumi_mesh_getNumEnt(m, 0);
  int num_own_vertex = pumi_mesh_getNumOwnEnt (m, 0);
  int num_elem = pumi_mesh_getNumEnt(m, mesh_dim);

  int num_dofs=6, num_values=1, node_elm = 3;
  if (mesh_dim==3) 
  {
    num_dofs=12;
    node_elm = 6;
  }

  int num_dofs_node = num_values * num_dofs;
  pField b_field = pumi_field_create (m, "b_field", num_dofs, PUMI_PACKED);
  pField c_field = pumi_field_create (m, "c_field", num_dofs, PUMI_PACKED);
  pField x_field = pumi_field_create (m, "x_field", num_dofs, PUMI_PACKED);

  if (!PCU_Comm_Self()) std::cout<<"* set b field ..."<<std::endl;

  pMeshEnt e;
  double xyz[3];
  pMeshIter it = m->begin(0);
  while ((e = m->iterate(it)))
  {
    pumi_node_getCoord(e, 0, xyz);
    // 2D mesh, z component =0
    std::vector<double> dofs(num_dofs*(1+scalar_type));
    for(int i=0; i<num_dofs_node*(1+scalar_type); ++i)
      dofs.at(i)=xyz[i%3];
    pumi_ment_setField(e, b_field, num_dofs_node, &dofs.at(0));
  }
  m->end(it);

//  PetscMemoryGetCurrentUsage(&mem);
//  PetscSynchronizedPrintf(MPI_COMM_WORLD, "process %d mem usage %f M \n ",PCU_Comm_Self(), mem/1e6);
//  PetscSynchronizedFlush(MPI_COMM_WORLD, NULL);
  if(!PCU_Comm_Self()) std::cout<<"* set matrix ..."<<std::endl; 
  t1 = MPI_Wtime();
  // fill matrix 
  // the matrix is diagnal dominant; thus should be positive definite
  int matrix_mult=1, matrix_solve=2;
  int matrix_mult_type = MSI_MULTIPLY;
  int matrix_solve_type = MSI_SOLVE;
  msi_matrix_create(matrix_mult, matrix_mult_type, b_field);
  msi_matrix_create(matrix_solve, matrix_solve_type, b_field);

  double diag_value=2.0, off_diag=1.0;
  int num_dofs_element = num_dofs_node*node_elm;
  std::vector<double> block(num_dofs_element*num_dofs_element*(1+scalar_type),0);
  for(int i=0; i<num_dofs_element; ++i)
  {
    for(int j=0; j<num_dofs_element; ++j)
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
  m->end(it);

  std::vector<pMeshEnt> adj_ents;
  it = m->begin(mesh_dim);
  while ((e = m->iterate(it)))
  {
    for(int rowVar=0; rowVar< num_values; ++rowVar)
    {
      for(int colVar=0; colVar< num_values; ++colVar)
      {
         std::vector<double> block_tmp = block;
         if(rowVar!=colVar)
           for(int i=0; i<block_tmp.size(); ++i) 
             block_tmp.at(i)*=0.5/num_values;
         msi_matrix_addBlock(matrix_solve, e, rowVar, colVar, &block_tmp[0]);
         msi_matrix_addBlock(matrix_mult, e, rowVar, colVar, &block_tmp[0]);
      }
    }
  }
  m->end(it);

  t2 = MPI_Wtime();
  //PetscMemoryGetCurrentUsage(&mem);
  //PetscSynchronizedPrintf(MPI_COMM_WORLD, "process %d mem usage %f M \n ",PCU_Comm_Self(), mem/1e6);
  //PetscSynchronizedFlush(MPI_COMM_WORLD,NULL);
  if(!PCU_Comm_Self()) std::cout<<"* assemble matrix ..."<<std::endl;
  msi_matrix_freeze(matrix_mult);
  msi_matrix_freeze(matrix_solve); 
  //msi_matrix_write(&matrix_mult, "matrixMult.m");
  //msi_matrix_write(&matrix_solve, "matrixSolve.m");
  // print out memory usage from petsc
  //PetscMemoryGetCurrentUsage(&mem);
  //PetscSynchronizedPrintf(MPI_COMM_WORLD, "process %d mem usage %f M \n ",PCU_Comm_Self(), mem/1e6);
  //PetscSynchronizedFlush(MPI_COMM_WORLD, NULL);
  t3 = MPI_Wtime();
  // calculate c field
  //pumi_field_print(&b_field);

  if(!PCU_Comm_Self()) std::cout<<"* do matrix-vector multiply ..."<<std::endl;
  msi_matrix_multiply(matrix_mult, b_field, c_field); 
  //pumi_field_print(&c_field);
  t4 = MPI_Wtime();
  // let's test field operations here
  pumi_field_copy(x_field, c_field); // c=x
  pumi_field_multiply(x_field, 2.0, x_field); // x = x*2.0
  pumi_field_add(x_field, c_field, x_field); // x = x+c

  it = m->begin(0);
  while ((e = m->iterate(it)))
  {
    std::vector<double> dofs1(num_dofs_node*(1+scalar_type)), dofs2(num_dofs_node*(1+scalar_type));
    pumi_ment_getField(e, c_field, 0, &dofs1.at(0));
    pumi_ment_getField(e, x_field, 0, &dofs2.at(0));
    for(int i=0; i<num_dofs_node*((1+scalar_type)); ++i)
      assert(AlmostEqualDoubles(dofs2.at(i), (2.0+1)*dofs1.at(i), 1e-6, 1e-6));
  }
  m->end(it);

  if(!PCU_Comm_Self()) std::cout<<"* solve ..."<<std::endl;
  // solve Ax=c
  int solver_type = 0;    // PETSc direct solver
  double solver_tol = 1e-6;
  msi_matrix_solve(matrix_solve, x_field, c_field); //, &solver_type, &solver_tol);
  //pumi_field_print(&x_field);
  t5 = MPI_Wtime();
  // verify x=b
  it = m->begin(0);
  while ((e = m->iterate(it)))
  {
    std::vector<double> dofs_x(num_dofs_node*(1+scalar_type)), dofs_b(num_dofs_node*(1+scalar_type));
    pumi_ment_getField(e, b_field, 0, &dofs_b.at(0));
    pumi_ment_getField(e, x_field, 0, &dofs_x.at(0));
    for(int idof=0; idof<num_dofs_node*(1+scalar_type); ++idof)
      assert(AlmostEqualDoubles(dofs_b.at(idof),dofs_x.at(idof), 1e-3, 1e-3));
  }
  m->end(it);

  msi_matrix_delete(matrix_mult);
  msi_matrix_delete(matrix_solve);
  if(!PCU_Comm_Self())
    std::cout<<"* time: fill matrix "<<t2-t1<<" assemble "<<t3-t2<<" mult "<<t4-t3<<" solve "<<t5-t4<<std::endl; 

  pumi_mesh_verify(m, false);

  pumi_field_delete(x_field);
  pumi_field_delete(b_field);
  pumi_field_delete(c_field);

  pumi_mesh_delete(m);

  PetscFinalize();
  pumi_finalize();
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

