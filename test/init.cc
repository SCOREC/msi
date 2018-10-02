#include <msi.h>
int main(int argc, char * argv[])
{
  MPI_Init(&argc,&argv);
  pumi_start();
  msi_init(argc,argv,MPI_COMM_WORLD);
  msi_finalize();
  MPI_Finalize();
  return 0;
}
