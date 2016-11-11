#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#define NROWS 4
#define NCOLS 4
#define MASTER 0
#define NX 4
#define NY 4

int main(int argc, char* argv[]){
  int *grid;

  int *temp1;
  int *temp2;

  int ii, jj;
  int iter;
  int size;

  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */

  int *sendbuf;       /* buffer to hold values to send */
  int *recvbuf;       /* buffer to hold received values */
  int *gridfinal;      /* buffer to hold values for printing */

  int tag = 0; /* scope for adding extra information to a message */

  int rank;
  int left;
  int right;

  MPI_Status status;

  grid= (int*)malloc(sizeof(int) * NX * NY);
  for (ii = 0; ii < NX; ii++){
    for(jj = 0; jj <NY; jj++){
      grid[ii*NX +jj] = ii;
    }
  }


  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  local_ncols = NCOLS;
  local_nrows = 1;
  if (rank == 0) gridfinal= (int*)malloc(sizeof(int) * NX * NY);
  temp1 = (int*)malloc(sizeof(int) * ((local_nrows+2)*local_ncols));
  //temp2 = (int*)malloc(sizeof(int) * ((local_nrows+2)*local_ncols));

  sendbuf = (int*)malloc(sizeof(int) * local_ncols);
  recvbuf = (int*)malloc(sizeof(int) * local_ncols);
  for (int it = 0; it < 3; it++){
    // devide grid between 4 threads
    for(ii=0;ii<local_nrows;ii++) {
      for(jj=0;jj<local_ncols;jj++) {
        temp1[(ii+1) * NX +jj] = grid[(ii+rank) * NX +jj];
      }
    }
    // copy data to be send left and right in this specific case

    for (ii = 0; ii<local_ncols;ii++){
      sendbuf[ii] = temp1[ NX +ii ];
    }

    // send data left and receive right

    MPI_Sendrecv(sendbuf,4,MPI_INT,left,tag,
                recvbuf,4,MPI_INT,right,tag,
              MPI_COMM_WORLD,&status);
    for (jj = 0; jj < local_ncols;jj++){
      temp1[(local_nrows +1)*NX +jj] = recvbuf[jj];
    }

    // send data right receive left
    MPI_Sendrecv(sendbuf,4,MPI_INT,right,tag,
                recvbuf,4,MPI_INT,left, tag,
                MPI_COMM_WORLD, &status);

    for (jj = 0; jj < local_ncols;jj++){
      temp1[jj] = recvbuf[jj];
    }

    // at this point all tanks have 3 rows so can do its calculation
    // do calculation

    for (jj = 0; jj <local_ncols;jj++){
      temp1[NX +jj] = temp1[NX +jj] + temp1[jj] +temp1[NX*2 +jj];
    }

    // join grid again
    if (rank == 0){
      for(jj =0; jj<local_ncols; jj++){
        printf("%3d ",temp1[NX +jj]);
        gridfinal[jj] = temp1[NX +jj];
        grid[jj] = temp1[NX +jj];
      }
      printf("\n");
      for(int k= 1; k <size;k++){
        MPI_Recv(recvbuf,4,MPI_INT,k,tag,MPI_COMM_WORLD,&status);
      	for(jj=0;jj < 4;jj++) {
      	  printf("%3d ",recvbuf[jj]);
          gridfinal[k*NX +jj] = recvbuf[jj];
          grid[k*NX +jj] = recvbuf[jj];
      	}
        printf("\n");
      }
      printf("\n");
    }
    else {
      for (ii = 0; ii<local_ncols;ii++){
        sendbuf[ii] = temp1[ NX +ii ];
      }
      MPI_Send(sendbuf,4,MPI_INT,0,tag,MPI_COMM_WORLD);
    }
}
  /* don't forget to tidy up whgedten we're done */
  MPI_Finalize();
  free(sendbuf);
  free(recvbuf);
  free(temp1);

  for (ii = 0; ii < NX; ii++){
    for(jj = 0; jj <NY; jj++){
      printf("%d\t",gridfinal[ii*NX +jj]);
    }
    printf("\n");
  }


}
