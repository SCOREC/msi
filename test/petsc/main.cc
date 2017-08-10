/****************************************************************************** 

  (c) 2005-2016 Scientific Computation Research Center, 
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
static char help[] = "testing solver functions; \n first do mat-vec product A*b=c; solve Ax=c; compare x and b\n\n";

bool AlmostEqualDoubles(double A, double B,
            double maxDiff, double maxRelDiff);

const char* modelFile = 0;
const char* meshFile = 0;

void getConfig(int argc, char** argv)
{
  if ( argc < 3 ) {
    if ( !PCU_Comm_Self() )
    {
      printf("Usage: %s <model> <distributed mesh>>\n", argv[0]);
      std::cout<<help;
    }
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
  if (argc<4 & !PCU_Comm_Self())
  {
    cout<<"Usage: ./main  model mesh #planes real(0)/complex(1) "<<endl;
    return MSI_FAILURE;
  }
  double t1, t2, t3, t4, t5;

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
  int num_vertex, num_own_vertex, vertex_dim=0;
  int num_elem, elem_dim=pumi_mesh_getDim(m);

  if(!PCU_Comm_Self()) cout<<"* Start MSI ..."<<endl;
  msi_start(m, NULL);

  num_vertex = pumi_mesh_getNumEnt(m, 0);
  num_own_vertex = pumi_mesh_getNumOwnEnt(m, 0); 
  num_elem = pumi_mesh_getNumEnt(m, elem_dim);


  int value_type[] = {scalar_type,scalar_type};

  int b_field=0, c_field=1, x_field=2;
  int num_values=1;

  int num_dofs=6;
  if (elem_dim==3) num_dofs=12;
  int num_dofs_node = num_values * num_dofs;
  if(!PCU_Comm_Self()) cout<<"* creating fields ..."<<endl;
  msi_field_create (&b_field, "b_field", &num_values, value_type, &num_dofs);
  msi_field_create (&c_field, "c_field", &num_values, value_type, &num_dofs);
  msi_field_create (&x_field, "x_field", &num_values, value_type, &num_dofs);

//  PetscMemoryGetCurrentUsage(&mem);
//  PetscSynchronizedPrintf(MPI_COMM_WORLD, "process %d mem usage %f M \n ",PCU_Comm_Self(), mem/1e6);
//  PetscSynchronizedFlush(MPI_COMM_WORLD, NULL);
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
    msi_ent_setdofdata(&vertex_dim, &inode, &b_field, &num_dofs_node, &dofs.at(0));
  }
//  PetscMemoryGetCurrentUsage(&mem);
//  PetscSynchronizedPrintf(MPI_COMM_WORLD, "process %d mem usage %f M \n ",PCU_Comm_Self(), mem/1e6);
//  PetscSynchronizedFlush(MPI_COMM_WORLD, NULL);



  t1 = MPI_Wtime();
  // fill matrix 
  // the matrix is diagnal dominant; thus should be positive definite
  int matrix_mult=1, matrix_solve=2;
  int matrix_mult_type = MSI_MULTIPLY;
  int matrix_solve_type = MSI_SOLVE;
  msi_matrix_create(&matrix_mult, &matrix_mult_type, value_type, &b_field);
  msi_matrix_create(&matrix_solve, &matrix_solve_type, value_type, &b_field);

  double diag_value=2.0, off_diag=1.0;
  int node_elm = 3;
  if(elem_dim==3) node_elm=6;
  int num_dofs_element = num_dofs_node*node_elm;
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
  std::cout<<"\n";

  for(int ielm = 0; ielm < num_elem; ielm++)
  {
    for(int rowVar=0; rowVar< num_values; rowVar++)
    {
      for(int colVar=0; colVar< num_values; colVar++)
      {
         vector<double> block_tmp = block;
         if (rowVar!=colVar)
         {
           for(int i=0; i<block_tmp.size(); i++) block_tmp.at(i)*=0.5/num_values;
         }
         msi_matrix_insertblock(&matrix_solve, &ielm, &rowVar, &colVar, &block_tmp[0]);
         msi_matrix_insertblock(&matrix_mult, &ielm, &rowVar, &colVar, &block_tmp[0]);
      }
    }
  }
  t2 = MPI_Wtime();
  //PetscMemoryGetCurrentUsage(&mem);
  //PetscSynchronizedPrintf(MPI_COMM_WORLD, "process %d mem usage %f M \n ",PCU_Comm_Self(), mem/1e6);
  //PetscSynchronizedFlush(MPI_COMM_WORLD,NULL);
  if(!PCU_Comm_Self()) cout<<"* assemble matrix ..."<<endl;
  msi_matrix_freeze(&matrix_mult);
  msi_matrix_freeze(&matrix_solve); 
  //msi_matrix_write(&matrix_mult, "matrixMult.m");
  //msi_matrix_write(&matrix_solve, "matrixSolve.m");
  // print out memory usage from petsc
  //PetscMemoryGetCurrentUsage(&mem);
  //PetscSynchronizedPrintf(MPI_COMM_WORLD, "process %d mem usage %f M \n ",PCU_Comm_Self(), mem/1e6);
  //PetscSynchronizedFlush(MPI_COMM_WORLD, NULL);
  t3 = MPI_Wtime();
  // calculate c field
  //msi_field_print(&b_field);

  if(!PCU_Comm_Self()) cout<<"* do matrix-vector multiply ..."<<endl;
  msi_matrix_multiply(&matrix_mult, &b_field, &c_field); 
  //msi_field_print(&c_field);
  t4 = MPI_Wtime();
  // let's test field operations here
 pumi_field_copy(pumi_mesh_getField(m, c_field), pumi_mesh_getField(m, x_field));
  double val[]={2.};
  int realtype=0;
  pumi_field_multiply(pumi_mesh_getField(m, x_field), 2.0, pumi_mesh_getField(m, x_field));
  pumi_field_add(pumi_mesh_getField(m, x_field), pumi_mesh_getField(m, c_field), pumi_mesh_getField(m, x_field));

  for(int i=0; i<num_vertex; i++)
  {
    vector<double> dofs1(num_dofs_node*(1+scalar_type)), dofs2(num_dofs_node*(1+scalar_type));
    int num_dofs_t;
    msi_ent_getdofdata(&vertex_dim, &i, &c_field, &num_dofs_t, &dofs1.at(0));
    assert(num_dofs_t==num_dofs_node);
    msi_ent_getdofdata(&vertex_dim, &i, &x_field, &num_dofs_t, &dofs2.at(0));
    assert(num_dofs_t==num_dofs_node);
    for(int i=0; i<num_dofs_node*((1+scalar_type)); i++)
      assert(AlmostEqualDoubles(dofs2.at(i), (val[0]+1)*dofs1.at(i), 1e-6, 1e-6));
  }
  // copy c field to x field
  pumi_field_copy(pumi_mesh_getField(m, c_field), pumi_mesh_getField(m, x_field));
  //msi_field_print(&x_field);
  if(!PCU_Comm_Self()) cout<<"* solve ..."<<endl;
  // solve Ax=c
  int solver_type = 0;    // PETSc direct solver
  double solver_tol = 1e-6;
  msi_matrix_solve(&matrix_solve, &x_field); //, &solver_type, &solver_tol);
  //msi_field_print(&x_field);
  t5 = MPI_Wtime();
  // verify x=b
  for(int inode=0; inode<num_vertex; inode++)
  {
    vector<double> dofs_x(num_dofs_node*(1+scalar_type)), dofs_b(num_dofs_node*(1+scalar_type));
    msi_ent_getdofdata(&vertex_dim, &inode, &b_field, &num_dofs_node, &dofs_b.at(0));
    msi_ent_getdofdata(&vertex_dim, &inode, &x_field, &num_dofs_node, &dofs_x.at(0));
    for(int idof=0; idof<num_dofs_node*(1+scalar_type); idof++)
      assert(AlmostEqualDoubles(dofs_b.at(idof),dofs_x.at(idof), 1e-3, 1e-3));
  }
  msi_matrix_delete(&matrix_mult);
  msi_matrix_delete(&matrix_solve);
  if(!PCU_Comm_Self())
    cout<<"* time: fill matrix "<<t2-t1<<" assemble "<<t3-t2<<" mult "<<t4-t3<<" solve "<<t5-t4<<endl; 

  pumi_mesh_verify(m, false);

  msi_field_delete(&x_field);
  msi_field_delete(&b_field);
  msi_field_delete(&c_field);

  PetscFinalize();
  msi_finalize(m);
  pumi_mesh_delete(m);
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

