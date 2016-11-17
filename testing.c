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
  int *temp2;

  int ii, jj;
  int iter;
  int size;

  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */

  t_speed *sendbuf;       /* buffer to hold values to send */
  t_speed *recvbuf;       /* buffer to hold received values */
  t_speed *gridfinal;      /* buffer to hold values for printing */

  int tag = 0; /* scope for adding extra information to a message */

  int rank;
  int left;
  int right;
  MPI_Datatype tspeed;
  MPI_Status status;

  printf("Original Grid\n");
  grid= (t_speed*)malloc(sizeof(t_speed) * NX * NY);
  for (ii = 0; ii < NX; ii++){
    for(jj = 0; jj <NY; jj++){
      for(int val = 0; val < 9; val++ ){
        grid[ii*NX +jj].speeds[val] = ii;
        printf("%3d ",ii);
      }
    }
    printf("\n");
  }
  printf("\n");


  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Type_contiguous(9, MPI_DOUBLE,&tspeed);
  MPI_Type_commit(&tspeed);

  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  local_ncols = NCOLS;
  local_nrows = 1;


  // allocate space for send and recv buffer
  sendbuf = (t_speed*)malloc(sizeof(t_speed) * local_ncols);
  recvbuf = (t_speed*)malloc(sizeof(t_speed) * local_ncols);

  if (rank == 0) gridfinal= (t_speed*)malloc(sizeof(t_speed) * NX * NY);
  temp1 = (t_speed*)malloc(sizeof(t_speed) * ((local_nrows+2)*local_ncols));
  //temp2 = (int*)malloc(sizeof(int) * ((local_nrows+2)*local_ncols));

  // devide grid between 4 threads
  for(ii=0;ii<local_nrows;ii++) {
    for(jj=0;jj<local_ncols;jj++) {
      for(int val = 0; val < 9; val++ ){
          temp1[(ii+1) * NX +jj].speeds[val] = grid[(ii+rank) * NX +jj].speeds[val];
      }
    }
  }

  // copy data to be send left and right in this specific case
  for (ii = 0; ii<local_ncols;ii++){
    for(int val = 0; val < 9; val++ ){
        sendbuf[ii].speeds[val]  = temp1[ NX +ii ].speeds[val] ;
    }
  }

  // send data left and receive right

  MPI_Sendrecv(sendbuf,local_ncols,tspeed,left,tag,
              recvbuf,local_ncols,tspeed,right,tag,
            MPI_COMM_WORLD,&status);

  for (jj = 0; jj < local_ncols;jj++){
    for(int val = 0; val < 9; val++ ){
      temp1[(local_nrows +1)*NX +jj].speeds[val] = recvbuf[jj].speeds[val] ;
    }
  }

  // send data right receive left
  MPI_Sendrecv(sendbuf,local_ncols,tspeed,right,tag,
              recvbuf,local_ncols,tspeed,left,tag,
            MPI_COMM_WORLD,&status);

  for (jj = 0; jj < local_ncols;jj++){
    for(int val = 0; val < 9; val++ ){
      temp1[jj].speeds[val] = recvbuf[jj].speeds[val] ;
    }
  }

  // at this point all tanks have 3 rows so can do its calculation
  // do calculation

  for (jj = 0; jj <local_ncols;jj++){
    for(int val = 0; val < 9; val++ ){
      temp1[NX +jj].speeds[val]  = temp1[NX +jj].speeds[val]  + temp1[jj].speeds[val] +temp1[NX*2 +jj].speeds[val] ;
    }
  }

  // join grid again
  if (rank == 0){
    for(jj =0; jj<local_ncols; jj++){
      for(int val = 0; val < 9; val++ ){
        printf("%3d ",temp1[NX +jj].speeds[val]);
        gridfinal[jj].speeds[val]  = temp1[NX +jj].speeds[val];
        grid[jj].speeds[val] = temp1[NX +jj].speeds[val];
      }
    }
    printf("\n");
    for(int k= 1; k <size;k++){
      MPI_Recv(recvbuf,local_ncols,tspeed,k,tag,MPI_COMM_WORLD,&status);
    	for(jj=0;jj < 4;jj++) {
        for(int val = 0; val < 9; val++ ){
  	     printf("%3d ",recvbuf[jj].speeds[val]);
         gridfinal[k*NX +jj].speeds[val]  = recvbuf[jj].speeds[val];
         grid[k*NX +jj].speeds[val] = recvbuf[jj].speeds[val];
        }
    	}
      printf("\n");
    }
    printf("\n");
  }
  else {
    for (ii = 0; ii<local_ncols;ii++){
      for(int val = 0; val < 9; val++ ){
        sendbuf[ii].speeds[val] = temp1[ NX +ii ].speeds[val];
      }
    }
    MPI_Send(sendbuf,local_ncols,tspeed,0,tag,MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  /* don't forget to tidy up whgedten we're done */
  MPI_Finalize();
  free(sendbuf);
  free(recvbuf);
  free(temp1);


}
