#include <msi.h>
int main(int argc, char * argv[])
{
  MPI_Init(&argc,&argv);
  int rnk = -1;
  MPI_Comm_rank(MPI_COMM_WORLD,&rnk);
  MPI_Comm nw;
  MPI_Comm_split(MPI_COMM_WORLD,rnk%2,rnk/2,&nw);
  pumi_start();
  msi_init(argc,argv,nw);
  msi_finalize();
  MPI_Finalize();
  return 0;
}
