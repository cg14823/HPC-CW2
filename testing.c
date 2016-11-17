#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#define NROWS 4
#define NCOLS 4
#define MASTER 0
#define NX 4
#define NY 4


typedef struct
{
  double speeds[9];
} t_speed;

int main(int argc, char* argv[]){
  t_speed *grid;

  t_speed *temp1;

  int ii, jj;
  int iter;
  int size;

  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */

  double *sendbuf;       /* buffer to hold values to send */
  double *recvbuf;       /* buffer to hold received values */
  t_speed *gridfinal;      /* buffer to hold values for printing */

  int tag = 0; /* scope for adding extra information to a message */

  int rank;
  int left;
  int right;
  int blocklen[1] = {9};
  MPI_Aint disp[1] ={0};


  MPI_Status status;

  grid= (t_speed*)malloc(sizeof(t_speed) * NX * NY);
  for (ii = 0; ii < NX; ii++){
    for(jj = 0; jj <NY; jj++){
      for(int val = 0; val < 9; val++ ){
        grid[ii*NX +jj].speeds[val] = ii;
      }
    }
  }


  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  local_ncols = NCOLS;
  local_nrows = 1;
  if (rank == 0){
    printf("ORIGINAL GIRD");
    for (ii = 0; ii < NX; ii++){
      for(jj = 0; jj <NY; jj++){
        for(int val = 0; val < 9; val++ ){
          printf("%.1f ",grid[ii*NX +jj].speeds[val]);
        }
      }
      printf("\n");
    }
    printf("\n");
  }

  // allocate space for send and recv buffer
  sendbuf = (double*)malloc(sizeof(double) * local_ncols*9);
  recvbuf = (double*)malloc(sizeof(double) * local_ncols*9);

  if (rank == 0) gridfinal= (t_speed*)malloc(sizeof(t_speed) * NX * NY);
  temp1 = (t_speed*)malloc(sizeof(t_speed) * ((local_nrows+2)*local_ncols));

  // devide grid between 4 threads
  for(ii=0;ii<local_nrows;ii++) {
    for(jj=0;jj<local_ncols;jj++) {
        temp1[(ii+1) * NX +jj] = grid[(ii+rank) * NX +jj];
    }
  }

  // copy data to be send left and right in this specific case
  for (ii = 0; ii<local_ncols;ii++){
    for(int val = 0; val < 9; val++ ){
        sendbuf[local_ncols* ii +val]  = temp1[ NX +ii ].speeds[val];
    }
  }

  // send data left and receive right

  MPI_Sendrecv(sendbuf,local_ncols*9,MPI_DOUBLE,left,tag,
              recvbuf,local_ncols*9,MPI_DOUBLE,right,tag,
            MPI_COMM_WORLD,&status);

  for (jj = 0; jj < local_ncols;jj++){
    for(int val = 0; val < 9; val++ ){
      temp1[(local_nrows +1)*NX +jj].speeds[val] = recvbuf[local_ncols * jj +val];
    }
  }

  // send data right receive left
  MPI_Sendrecv(sendbuf,local_ncols*9,MPI_DOUBLE,right,tag,
              recvbuf,local_ncols*9,MPI_DOUBLE,left,tag,
            MPI_COMM_WORLD,&status);

  for (jj = 0; jj < local_ncols;jj++){
    for(int val = 0; val < 9; val++ ){
      temp1[jj].speeds[val] = recvbuf[local_ncols* jj+val];
    }
  }

  // at this point all tanks have 3 rows so can do its calculation
  // do calculation

  for (jj = 0; jj <local_ncols;jj++){
    for(int val = 0; val < 9; val++ ){
      temp1[NX +jj].speeds[val]  += temp1[jj].speeds[val] +temp1[NX*2 +jj].speeds[val] ;
    }
  }

  // join grid again
  if (rank == 0){
    for(jj =0; jj<local_ncols; jj++){
      gridfinal[jj] = temp1[NX +jj];
      grid[jj]= temp1[NX +jj];
      for(int val = 0; val < 9; val++ ){
        printf("%.1f ",temp1[NX +jj].speeds[val]);
      }
    }
    printf("\n");
    for(int k= 1; k <size;k++){
      MPI_Recv(recvbuf,local_ncols*9,MPI_DOUBLE,k,tag,MPI_COMM_WORLD,&status);
    	for(jj=0;jj < 4;jj++) {
        for(int val = 0; val < 9; val++ ){
  	     printf("%.1f ",recvbuf[local_ncols* jj+val]);
         gridfinal[k*NX +jj].speeds[val]  = recvbuf[local_ncols* jj +val];
         grid[k*NX +jj].speeds[val] = recvbuf[local_ncols* jj +val];
        }
    	}
      printf("\n");
    }
    printf("\n");
  }
  else {
    for (ii = 0; ii<local_ncols;ii++){
      for(int val = 0; val < 9; val++ ){
        sendbuf[local_ncols* ii +val] = temp1[ NX +ii ].speeds[val];
      }
    }
    MPI_Send(sendbuf,local_ncols*9,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  /* don't forget to tidy up whgedten we're done */
  MPI_Finalize();
  free(sendbuf);
  free(recvbuf);
  free(temp1);

}
