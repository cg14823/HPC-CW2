
/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny

**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include "mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define MASTER 0
#define CHUNK 32

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int local_nrows);
int propagate(const t_param params, t_speed* partial_cells, t_speed* partial_temp_cells, int local_nrows,t_speed* top_halo, t_speed* bottom_halo);
int collisionrebound(const t_param params, t_speed* partial_cells, t_speed* partial_temp_cells, int* obstacles,int local_ncols, int local_nrows,int rank);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);
int halo_exchange(t_speed* partial_cells,int local_ncols,int local_nrows, float* sendgrid, float* recvgrid, int left, int right, int rank, t_speed* top_halo, t_speed* bottom_halo);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);
int calc_nrows_from_rank(int rank, int size, int ny);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
  int ii, jj,i;
  int rank;
  int left;
  int right;
  int size;
  int val;
  int local_ncols = params.nx;      // devide the grid by rows

  MPI_Status status;

  float *sendgrid;
  float *recvgrid;

  float *sendbufFINAL;
  float *recvbufFINAL;

  t_speed *partial_cells;
  t_speed *partial_temp_cells;
  t_speed *top_halo;
  t_speed *bottom_halo;

  int tag = 0; /* scope for adding extra information to a message */
  /* iterate for maxIters timesteps */

  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );


  // DEVIDE GRID DEFENETLY NOT BEST WAY
  left = (rank == MASTER) ? (size - 1) : (rank - 1);
  right = (rank + 1) % size;

  int local_nrows = calc_nrows_from_rank(rank,size,params.ny);       // all possibility nicely divicble by 64

  partial_cells = (t_speed*)malloc(sizeof(t_speed) * local_ncols * local_nrows );
  partial_temp_cells = (t_speed*)malloc(sizeof(t_speed) * local_ncols * local_nrows);

  top_halo = (t_speed*)malloc(sizeof(t_speed) * local_ncols);
  bottom_halo = (t_speed*)malloc(sizeof(t_speed) * local_ncols);

  for (ii = 0; ii< local_nrows;ii++){
    for(jj = 0; jj<local_ncols;jj++){
      partial_cells[ii* params.nx +jj] =  cells[(ii +rank*local_nrows) * params.nx+jj];
      partial_temp_cells[ii * params.nx +jj] = tmp_cells[(ii +rank*local_nrows) *params.nx+jj];
    }
  }
  for(jj=0;jj<local_ncols;jj++){
    if (rank == size-1){
      top_halo[jj] = cells[jj];
      bottom_halo[jj] = cells[(rank * local_nrows-1) * params.nx+jj];
    }
    else if (rank == MASTER){
      top_halo[jj] = cells[(local_nrows +rank*local_nrows) * params.nx+jj];
      bottom_halo[jj] = cells[(size * local_nrows-1) * params.nx+jj];
    }
    else{
      top_halo[jj] = cells[(local_nrows +rank*local_nrows) * params.nx+jj];
      bottom_halo[jj] = cells[(rank * local_nrows-1) * params.nx+jj];
    }

  }

  sendgrid = (float*)malloc(sizeof(float) * CHUNK * NSPEEDS);
  recvgrid = (float*)malloc(sizeof(float) * CHUNK * NSPEEDS);

  if (rank == MASTER){
    gettimeofday(&timstr, NULL);
    tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    //if (size != 64) printf("DAMMMMMMMMM\n");
  }

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    //  if (rank == MASTER) printf("it %d\n",tt);
    // !!!!------------------------------------HALO EXCHANGE --------------------------------------------------------!!!!
    if (rank == size - 1) accelerate_flow(params, partial_cells, obstacles,local_nrows);
    halo_exchange(partial_cells,local_ncols, local_nrows, sendgrid, recvgrid, left,  right, rank,top_halo,bottom_halo);
    propagate(params, partial_cells, partial_temp_cells,local_nrows,top_halo,bottom_halo);
    collisionrebound(params,partial_cells,partial_temp_cells,obstacles,local_ncols, local_nrows,rank);

    // START av_velocity
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u = 0.0f;          /* accumulated magnitudes of velocity for each cell */
    /* initialise */
    /* loop over all non-blocked cells */
    for (ii = 0; ii < local_nrows; ii++)
    {
      for (jj = 0; jj < params.nx; jj++)
      {
        /* ignore occupied cells */
        if (!obstacles[(ii+rank*local_nrows)*params.nx+jj])
        {
          int cellAccess = ii * params.nx + jj;
          /* local density total */
          float local_density = 0.0f;

          for (int kk = 0; kk < NSPEEDS; kk++)
          {
            local_density += partial_cells[cellAccess].speeds[kk];
          }

          /* x-component of velocity */
          float u_x = (partial_cells[cellAccess].speeds[1]
                        + partial_cells[cellAccess].speeds[5]
                        + partial_cells[cellAccess].speeds[8]
                        - (partial_cells[cellAccess].speeds[3]
                           + partial_cells[cellAccess].speeds[6]
                           + partial_cells[cellAccess].speeds[7]));
          /* compute y velocity component */
          float u_y = (partial_cells[cellAccess].speeds[2]
                        + partial_cells[cellAccess].speeds[5]
                        + partial_cells[cellAccess].speeds[6]
                        - (partial_cells[cellAccess].speeds[4]
                           + partial_cells[cellAccess].speeds[7]
                           + partial_cells[cellAccess].speeds[8]));
          /* accumulate the norm of x- and y- velocity components */
          tot_u += sqrt((u_x * u_x) + (u_y * u_y))/local_density;
          /* increase counter of inspected cells */
          tot_cells += 1;
        }
      }
    }
    float vars [2] = {tot_u,(float)tot_cells};
    float global[2]= {0.0f,0.0f};
    MPI_Reduce(&vars, &global, 2, MPI_FLOAT, MPI_SUM,MASTER, MPI_COMM_WORLD);

    if (rank == MASTER){
      av_vels[tt] = global[0]/global[1];
      //printf("==timestep: %d==\n", tt);
      //printf("av velocity: %.12E\n",av_vels[tt]);
      //printf("global[1]: %.12E\n",global[1]);
    }
  }
  free(sendgrid);
  free(recvgrid);
  if(rank == MASTER){
    gettimeofday(&timstr, NULL);
    toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr = ru.ru_utime;
    usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    timstr = ru.ru_stime;
    systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    // join grid

    //printf("start\n");
    recvbufFINAL  = (float*)malloc(sizeof(float)*CHUNK *NSPEEDS);
    for (int k = 1; k < size; k++){
      int rows = calc_nrows_from_rank(k,size,params.ny);
      for(ii = 0;ii<rows;ii++){
        for(jj=0;jj<local_ncols;jj+=CHUNK){
          MPI_Recv(recvbufFINAL,CHUNK*NSPEEDS,MPI_FLOAT,k,tag,MPI_COMM_WORLD,&status);
          for(int x =0;x<CHUNK;x++){
            for(int val =0; val <NSPEEDS; val++){
              cells[(k*local_nrows+ii)*params.nx+jj+x].speeds[val] = recvbufFINAL[x * NSPEEDS +val];
            }
          }
        }
      }
      //printf("end receving from %d\n",k);
    }
    free(recvbufFINAL);
    for (ii =0 ; ii<local_nrows;ii++){
      for (jj= 0;jj < local_ncols;jj++){
        cells[ii*params.nx +jj] = partial_cells[ii*params.nx+jj];
      }
    }

    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);

    write_values(params, cells, obstacles, av_vels);

  }
  else{
    sendbufFINAL  = (float*)malloc(sizeof(float) * CHUNK *NSPEEDS);
    for(ii =0;ii<local_nrows;ii++){
      for(jj=0;jj<local_ncols;jj += CHUNK){
        for(int x =0;x<CHUNK;x++){
          for(int val =0; val <NSPEEDS; val++){
            sendbufFINAL[x*NSPEEDS+val] = partial_cells[ii*params.nx +jj+x].speeds[val];
          }
        }
        MPI_Send(sendbufFINAL,CHUNK*NSPEEDS,MPI_FLOAT,MASTER,tag,MPI_COMM_WORLD);
      }
    }
    free(sendbufFINAL);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  free(partial_temp_cells);
  free(partial_cells);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
//  accelerate_flow(params, cells, obstacles);
  //propagate(params, cells, tmp_cells);
//collisionrebound(params, cells, tmp_cells, obstacles);
  return EXIT_SUCCESS;
}

int halo_exchange(t_speed* partial_cells,int local_ncols,int local_nrows, float* sendgrid, float* recvgrid, int left, int right, int rank,t_speed* top_halo, t_speed* bottom_halo){
  // copy data to be send left 1st row
  MPI_Status status;
  int tag =0;
  int chunksize = CHUNK*NSPEEDS;

  for (int jj = 0; jj<local_ncols;jj+=CHUNK){
    // send first row left and receive row from right  to put on top
    for(int x = 0; x<CHUNK;x++){
      for(int val = 0; val<NSPEEDS; val++){
        sendgrid[x*NSPEEDS +val] = partial_cells[jj+x].speeds[val];
      }
    }
    MPI_Sendrecv(sendgrid,chunksize,MPI_FLOAT,left,tag,
                recvgrid,chunksize,MPI_FLOAT,right,tag,
                MPI_COMM_WORLD,&status);
    for(int x = 0; x<CHUNK;x++){
      for(int val = 0; val<NSPEEDS; val++){
        top_halo[jj+x].speeds[val] = recvgrid[x*NSPEEDS+val];
      }
    }

  }

  for (int jj = 0; jj<local_ncols;jj+=CHUNK){
    // send first row left and receive row from right  to put on top
    for(int x = 0; x<CHUNK;x++){
      for(int val = 0; val<NSPEEDS; val++){
        sendgrid[x*NSPEEDS +val] = partial_cells[(local_nrows-1)*local_ncols+jj+x].speeds[val];
      }
    }
    MPI_Sendrecv(sendgrid,chunksize,MPI_FLOAT,right,tag,
                recvgrid,chunksize,MPI_FLOAT,left,tag,
                MPI_COMM_WORLD,&status);
    for(int x = 0; x<CHUNK;x++){
      for(int val = 0; val<NSPEEDS; val++){
        bottom_halo[jj+x].speeds[val] = recvgrid[x*NSPEEDS+val];
      }
    }

  }


  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* partial_cells, int* obstacles, int local_nrows)
{
  float aw1 = params.density * params.accel / 9.0f;
  float aw2 = params.density * params.accel / 36.0f;


  /* modify the 2nd row of the grid */
  int ii = local_nrows -2;

  for (int jj = 0; jj < params.nx; jj++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[(params.ny -2)* params.nx + jj]
        && (partial_cells[ii * params.nx + jj].speeds[3] - aw1) > 0.0f
        && (partial_cells[ii * params.nx + jj].speeds[6] - aw2) > 0.0f
        && (partial_cells[ii * params.nx + jj].speeds[7] - aw2) > 0.0f)
    {
      /* increase 'east-side' densities */
      partial_cells[ii * params.nx + jj].speeds[1] += aw1;
      partial_cells[ii * params.nx + jj].speeds[5] += aw2;
      partial_cells[ii * params.nx + jj].speeds[8] += aw2;
      /* decrease 'west-side' densities */
      partial_cells[ii * params.nx + jj].speeds[3] -= aw1;
      partial_cells[ii * params.nx + jj].speeds[6] -= aw2;
      partial_cells[ii * params.nx + jj].speeds[7] -= aw2;
    }
  }
  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* partial_cells, t_speed* partial_temp_cells, int local_nrows, t_speed* top_halo, t_speed* bottom_halo)
{
  for (int ii = 0; ii < local_nrows; ii++)
  {
    int y_n = ii + 1;
    int y_s = ii - 1;
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */

      int x_e = (jj + 1) % params.nx;
      int x_w = (jj == 0) ? ( params.nx - 1) : (jj - 1);
      /* propagate densities to neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      if( ii == 0){
        partial_temp_cells[ii* params.nx + jj].speeds[0] = partial_cells[ii * params.nx + jj].speeds[0]; /* central cell, no movement */
        partial_temp_cells[ii* params.nx + jj].speeds[1] = partial_cells[ii * params.nx + x_w].speeds[1]; /* east */
        partial_temp_cells[ii* params.nx + jj].speeds[2] = bottom_halo[jj].speeds[2]; /* north */
        partial_temp_cells[ii* params.nx + jj].speeds[3] = partial_cells[ii * params.nx + x_e].speeds[3]; /* west */
        partial_temp_cells[ii* params.nx + jj].speeds[4] = partial_cells[y_n * params.nx + jj].speeds[4]; /* south */
        partial_temp_cells[ii* params.nx + jj].speeds[5] = bottom_halo[x_w].speeds[5]; /* north-east */
        partial_temp_cells[ii* params.nx + jj].speeds[6] = bottom_halo[x_e].speeds[6]; /* north-west */
        partial_temp_cells[ii* params.nx + jj].speeds[7] = partial_cells[y_n * params.nx + x_e].speeds[7]; /* south-west */
        partial_temp_cells[ii* params.nx + jj].speeds[8] = partial_cells[y_n * params.nx + x_w].speeds[8]; /* south-east */

      }
      else if(ii == local_nrows - 1){
        partial_temp_cells[ii* params.nx + jj].speeds[0] = partial_cells[ii * params.nx + jj].speeds[0]; /* central cell, no movement */
        partial_temp_cells[ii* params.nx + jj].speeds[1] = partial_cells[ii * params.nx + x_w].speeds[1]; /* east */
        partial_temp_cells[ii* params.nx + jj].speeds[2] = partial_cells[y_s * params.nx + jj].speeds[2]; /* north */
        partial_temp_cells[ii* params.nx + jj].speeds[3] = partial_cells[ii * params.nx + x_e].speeds[3]; /* west */
        partial_temp_cells[ii* params.nx + jj].speeds[4] = top_halo[jj].speeds[4]; /* south */
        partial_temp_cells[ii* params.nx + jj].speeds[5] = partial_cells[y_s * params.nx + x_w].speeds[5]; /* north-east */
        partial_temp_cells[ii* params.nx + jj].speeds[6] = partial_cells[y_s * params.nx + x_e].speeds[6]; /* north-west */
        partial_temp_cells[ii* params.nx + jj].speeds[7] = top_halo[x_e].speeds[7]; /* south-west */
        partial_temp_cells[ii* params.nx + jj].speeds[8] = top_halo[x_w].speeds[8]; /* south-east */

      }
      else{
        partial_temp_cells[ii* params.nx + jj].speeds[0] = partial_cells[ii * params.nx + jj].speeds[0]; /* central cell, no movement */
        partial_temp_cells[ii* params.nx + jj].speeds[1] = partial_cells[ii * params.nx + x_w].speeds[1]; /* east */
        partial_temp_cells[ii* params.nx + jj].speeds[2] = partial_cells[y_s * params.nx + jj].speeds[2]; /* north */
        partial_temp_cells[ii* params.nx + jj].speeds[3] = partial_cells[ii * params.nx + x_e].speeds[3]; /* west */
        partial_temp_cells[ii* params.nx + jj].speeds[4] = partial_cells[y_n * params.nx + jj].speeds[4]; /* south */
        partial_temp_cells[ii* params.nx + jj].speeds[5] = partial_cells[y_s * params.nx + x_w].speeds[5]; /* north-east */
        partial_temp_cells[ii* params.nx + jj].speeds[6] = partial_cells[y_s * params.nx + x_e].speeds[6]; /* north-west */
        partial_temp_cells[ii* params.nx + jj].speeds[7] = partial_cells[y_n * params.nx + x_e].speeds[7]; /* south-west */
        partial_temp_cells[ii* params.nx + jj].speeds[8] = partial_cells[y_n * params.nx + x_w].speeds[8]; /* south-east */
      }


    }
  }

  return EXIT_SUCCESS;
}

int calc_nrows_from_rank(int rank, int size, int ny)
{
  int nrows;

  nrows = ny / size;       /* integer division */
  if ((ny % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      nrows += ny % size;  /* add remainder to last rank */
  }

  return nrows;
}

int collisionrebound(const t_param params, t_speed* partial_cells, t_speed* partial_temp_cells, int* obstacles,int local_ncols, int local_nrows,int rank)
{
  const float w0 = 4.0f / 9.0f;  /* weighting factor */
  const float w1 = 1.0f / 9.0f;  /* weighting factor */
  const float w2 = 1.0f / 36.0f; /* weighting factor */
  int ii,jj;

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */

  for (ii = 0; ii < local_nrows; ii++)
  {
    for (jj = 0; jj < params.nx; jj++)
    {
      int cellAccess = ii * params.nx + jj;
      /* don't consider occupied cells */
      if (!obstacles[(ii+rank*local_nrows)*params.nx+jj])
      {

        /* compute local density total */
        float local_density = 0.0f;
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += partial_temp_cells[cellAccess].speeds[kk];
        }
        /* compute x velocity component */
        float u_x = (partial_temp_cells[cellAccess].speeds[1]
                      + partial_temp_cells[cellAccess].speeds[5]
                      + partial_temp_cells[cellAccess].speeds[8]
                      - (partial_temp_cells[cellAccess].speeds[3]
                         + partial_temp_cells[cellAccess].speeds[6]
                         + partial_temp_cells[cellAccess].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (partial_temp_cells[cellAccess].speeds[2]
                      + partial_temp_cells[cellAccess].speeds[5]
                      + partial_temp_cells[cellAccess].speeds[6]
                      - (partial_temp_cells[cellAccess].speeds[4]
                         + partial_temp_cells[cellAccess].speeds[7]
                         + partial_temp_cells[cellAccess].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;
        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density * (1.0f - 1.5 * u_sq);
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.0f + 3.0f * (u_x + u_x * u_x) - 1.5f * u_y * u_y);
        d_equ[2] = w1 * local_density * (1.0f + 3.0f * (u_y + u_y * u_y) - 1.5f * u_x * u_x);
        d_equ[3] = w1 * local_density * (1.0f + 3.0f * (-u_x + u_x * u_x) - 1.5f * u_y * u_y);
        d_equ[4] = w1 * local_density * (1.0f + 3.0f * (-u_y + u_y * u_y) - 1.5f * u_x *u_x);
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.0f + 3.0f * (u_sq + u_x + u_y) + 9.0f * u_x * u_y);
        d_equ[6] = w2 * local_density * (1.0f + 3.0f * (u_sq - u_x + u_y) - 9.0f * u_x * u_y);
        d_equ[7] = w2 * local_density * (1.0f + 3.0f * (u_sq - u_x - u_y) + 9.0f * u_x * u_y);
        d_equ[8] = w2 * local_density * (1.0f + 3.0f * (u_sq + u_x - u_y) - 9.0f * u_x * u_y);

        /* relaxation step */

        for (int kk = 0; kk < NSPEEDS; kk++)
          {
            partial_cells[cellAccess].speeds[kk] = partial_temp_cells[cellAccess].speeds[kk]
                                                    + params.omega
                                                    * (d_equ[kk] - partial_temp_cells[cellAccess].speeds[kk]);
          }
      }
      else{
        partial_cells[cellAccess].speeds[1] = partial_temp_cells[cellAccess].speeds[3];
        partial_cells[cellAccess].speeds[2] = partial_temp_cells[cellAccess].speeds[4];
        partial_cells[cellAccess].speeds[3] = partial_temp_cells[cellAccess].speeds[1];
        partial_cells[cellAccess].speeds[4] = partial_temp_cells[cellAccess].speeds[2];
        partial_cells[cellAccess].speeds[5] = partial_temp_cells[cellAccess].speeds[7];
        partial_cells[cellAccess].speeds[6] = partial_temp_cells[cellAccess].speeds[8];
        partial_cells[cellAccess].speeds[7] = partial_temp_cells[cellAccess].speeds[5];
        partial_cells[cellAccess].speeds[8] = partial_temp_cells[cellAccess].speeds[6];
      }
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.0f;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  /* loop over all non-blocked cells */
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
        int cellAccess = ii * params.nx + jj;
        /* local density total */
        float local_density = 0.0f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[cellAccess].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[cellAccess].speeds[1]
                      + cells[cellAccess].speeds[5]
                      + cells[cellAccess].speeds[8]
                      - (cells[cellAccess].speeds[3]
                         + cells[cellAccess].speeds[6]
                         + cells[cellAccess].speeds[7]));
        /* compute y velocity component */
        float u_y = (cells[cellAccess].speeds[2]
                      + cells[cellAccess].speeds[5]
                      + cells[cellAccess].speeds[6]
                      - (cells[cellAccess].speeds[4]
                         + cells[cellAccess].speeds[7]
                         + cells[cellAccess].speeds[8]));
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y))/local_density;
        /* increase counter of inspected cells */
        tot_cells += 1;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.0f / 9.0f;
  float w1 = params->density      / 9.0f;
  float w2 = params->density      / 36.0f;

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.0f / 6.0f * (2.0f / params.omega - 1.0f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.0f;  /* accumulator */

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii * params.nx + jj].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.0f / 3.0f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii * params.nx + jj])
      {
        u_x = u_y = u = 0.0f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii * params.nx + jj].speeds[1]
               + cells[ii * params.nx + jj].speeds[5]
               + cells[ii * params.nx + jj].speeds[8]
               - (cells[ii * params.nx + jj].speeds[3]
                  + cells[ii * params.nx + jj].speeds[6]
                  + cells[ii * params.nx + jj].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii * params.nx + jj].speeds[2]
               + cells[ii * params.nx + jj].speeds[5]
               + cells[ii * params.nx + jj].speeds[6]
               - (cells[ii * params.nx + jj].speeds[4]
                  + cells[ii * params.nx + jj].speeds[7]
                  + cells[ii * params.nx + jj].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
