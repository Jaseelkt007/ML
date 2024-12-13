#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                   OPTIC FLOW USING HORN AND SCHUNCK                      */
/*                                                                          */
/*                 (Copyrights Joachim Weickert, 2/2003)                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/


/* 
 features:
 - Jacobi scheme for solving the Euler--Lagrange equations
*/


/*--------------------------------------------------------------------------*/

void alloc_vector

     (float **vector,   /* vector */
      long  n)          /* size */

     /* allocates storage for a vector of size n */


{
*vector = (float *) malloc (n * sizeof(float));
if (*vector == NULL)
   {
   printf("alloc_vector: not enough storage available\n");
   exit(1);
   }
return;
}

/*--------------------------------------------------------------------------*/

void alloc_matrix

     (float ***matrix,  /* matrix */
      long  nx,         /* size in x direction */
      long  ny)         /* size in y direction */

     /* allocates storage for matrix of size nx * ny */


{
long i;

*matrix = (float **) malloc (nx * sizeof(float *));
if (*matrix == NULL)
   {
   printf("alloc_matrix: not enough storage available\n");
   exit(1);
   }
for (i=0; i<nx; i++)
    {
    (*matrix)[i] = (float *) malloc (ny * sizeof(float));
    if ((*matrix)[i] == NULL)
       {
       printf("alloc_matrix: not enough storage available\n");
       exit(1);
       }
    }
return;
}

/*--------------------------------------------------------------------------*/

void disalloc_vector

     (float *vector,    /* vector */
      long  n)          /* size */

     /* disallocates storage for a vector of size n */

{
free(vector);
return;
}

/*--------------------------------------------------------------------------*/

void disalloc_matrix

     (float **matrix,   /* matrix */
      long  nx,         /* size in x direction */
      long  ny)         /* size in y direction */

     /* disallocates storage for matrix of size nx * ny */

{
long i;
for (i=0; i<nx; i++)
    free(matrix[i]);
free(matrix);
return;
}

/*--------------------------------------------------------------------------*/

void dummies

     (float **v,        /* image matrix */
      long  nx,         /* size in x direction */
      long  ny)         /* size in y direction */

/* creates dummy boundaries by mirroring */

{
long i, j;  /* loop variables */

for (i=1; i<=nx; i++)
    {
    v[i][0]    = v[i][1];
    v[i][ny+1] = v[i][ny];
    }

for (j=0; j<=ny+1; j++)
    {
    v[0][j]    = v[1][j];
    v[nx+1][j] = v[nx][j];
    }
return;
}

/* ---------------------------------------------------------------------- */

void flow 

     (long     nx,          /* image dimension in x direction */ 
      long     ny,          /* image dimension in y direction */ 
      float    hx,          /* pixel size in x direction */
      float    hy,          /* pixel size in y direction */
      float    **fx,        /* x derivative of image */
      float    **fy,        /* y derivative of image */
      float    **fz,        /* z derivative of image */
      float    alpha,       /* smoothness weight */
      float    **u,         /* x component of optic flow */
      float    **v)         /* v component of optic flow */


/* 
 Performs one Jacobi iteration for the Euler-Lagrange equations
 arising from the Horn and Schunck method. 
*/

{
long    i, j;             /* loop variables */
long    nn;               /* number of neighbours */
float   help;             /* 1.0/alpha */
float   **u1, **v1;       /* u, v at old iteration level */
      

/* ---- allocate storage ---- */

alloc_matrix (&u1, nx+2, ny+2);
alloc_matrix (&v1, nx+2, ny+2);


/* ---- copy u, v into u1, v1 ---- */

for (i=1; i<=nx; i++){
 for (j=1; j<=ny; j++)
     {
     u1[i][j] = u[i][j];
     v1[i][j] = v[i][j];
     }
}

/* ----Perform one Jacobi iteration*/
for (i =1;i<=nx;i++){
     for (j=1;j<=ny;j++){
          float sum_u_neighbors = 0.0;
          float sum_v_neighbors = 0.0;
          nn = 0;

          if (i>1){
               /*Add the top neighbor*/
               sum_u_neighbors += u1[i-1][j];
               sum_v_neighbors += v1[i-1][j];
               nn += 1;
          }
          if (i<nx){// Bottom neighbor
               sum_u_neighbors += u1[i+1][j];
               sum_v_neighbors += v1[i+1][j];
               nn += 1;
          }
          if (j>1){// Left neighbor
               sum_u_neighbors += u1[i][j-1];
               sum_v_neighbors += v1[i][j-1];
               nn += 1;
          }
          if (j<ny){ // Right neighbor
               sum_u_neighbors += u1[i][j+1];
               sum_v_neighbors += v1[i][j+1];
               nn += 1;
          }
          float denominator_u = fx[i][j] * fx[i][j] + alpha * nn;
          float denominator_v = fy[i][j] * fy[i][j] + alpha * nn;

          if (denominator_u != 0){
          u[i][j] = ((-fx[i][j]*(fy[i][j] * v1[i][j]) + fz[i][j]) + (alpha * sum_u_neighbors)) / denominator_u;
          }
          if (denominator_v != 0){
          v[i][j] = ((-fy[i][j]*(fx[i][j] * u1[i][j]) + fz[i][j]) + (alpha * sum_v_neighbors)) / denominator_v ;
          }
     }
}     

/* ---- disallocate storage ---- */

disalloc_matrix (u1, nx+2, ny+2);
disalloc_matrix (v1, nx+2, ny+2);
return;

} /* flow */

/* ---------------------------------------------------------------------- */

void analyse

     (float   **u,         /* image, unchanged */
      long    nx,          /* pixel number in x direction */
      long    ny,          /* pixel number in x direction */
      float   *min,        /* minimum, output */
      float   *max,        /* maximum, output */
      float   *mean,       /* mean, output */
      float   *vari)       /* variance, output */

/*
 calculates minimum, maximum, mean and variance of an image u
*/

{
long    i, j;       /* loop variables */
float   help;       /* auxiliary variable */
double  help2;      /* auxiliary variable */

*min  = u[1][1];
*max  = u[1][1];
help2 = 0.0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     if (u[i][j] < *min) *min = u[i][j];
     if (u[i][j] > *max) *max = u[i][j];
     help2 = help2 + (double)u[i][j];
     }
*mean = (float)help2 / (nx * ny);

*vari = 0.0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     help  = u[i][j] - *mean;
     *vari = *vari + help * help;
     }
*vari = *vari / (nx * ny);

return;

} /* analyse */

/*--------------------------------------------------------------------------*/

int main ()

{
char   row[80];              /* for reading data */
char   in1[80], in2[80];     /* for reading data */
char   out[80];              /* for reading data */
float  **f1, **f2;           /* images */
float  **fx, **fy, **fz;     /* image derivatives */
float  **u, **v;             /* optic flow components */
float  **w;                  /* optic flow magnitude */
long   i, j, k;              /* loop variables */
long   kmax;                 /* max. no. of iterations */
long   nx, ny;               /* image size in x, y direction */
FILE   *inimage, *outimage;  /* input file, output file */
float  hx, hy;               /* pixel sizes */
float  alpha;                /* smoothness weight */
float  max, min;             /* largest, smallest grey value */
float  mean;                 /* average grey value */
float  vari;                 /* variance */
float  help;                 /* time saver */
unsigned char byte;          /* for data conversion */


printf("\n");
printf("OPTIC FLOW COMPUTATION WITH THE METHOD OF HORN AND SCHUNCK\n\n");
printf("***********************************************\n\n");
printf("  2/2003 Copyright by Joachim Weickert         \n");
printf("  Faculty of Mathematics and Computer Science  \n");
printf("  Saarland University, Saarbruecken, Germany.  \n\n");
printf("  All rights reserved. Unauthorized usage,     \n");
printf("  copying, hiring, and selling prohibited.     \n\n");
printf("  Send bug reports to                          \n");
printf("  bruhn@vis.uni-stuttgart.de                   \n\n");
printf("***********************************************\n\n");


/* ---- read input image f1 (pgm format P5) ---- */

/* read image name */
printf("input image 1:                    ");
/*gets (in1);*/
/* open pgm file and read header */
inimage = fopen("/workspaces/ML/Computer_vision_Topics/Horn_and_Schunk_optical_flow/pig1.pgm","rb");
if (inimage == NULL ){
     perror("Error opening file1");
     exit(1);
}

fgets (row, 80, inimage);
fgets (row, 80, inimage);
while (row[0]=='#') fgets(row, 80, inimage);
sscanf (row, "%ld %ld", &nx, &ny);
fgets (row, 80, inimage);

/* allocate storage */
alloc_matrix (&f1, nx+2, ny+2);

/* read image data */
for (j=1; j<=ny; j++)
 for (i=1; i<=nx; i++)
     f1[i][j] = (float) getc (inimage);
fclose(inimage);


/* ---- read input image f2 (pgm format P5) ---- */

/* read image name */
printf("input image 2:                    ");
inimage = fopen("/workspaces/ML/Computer_vision_Topics/Horn_and_Schunk_optical_flow/pig2.pgm","rb");
if (inimage == NULL ){
     perror("Error opening file2");
     exit(1);
}
/* open pgm file and read header */
fgets (row, 80, inimage);
fgets (row, 80, inimage);
while (row[0]=='#') fgets(row, 80, inimage);
sscanf (row, "%ld %ld", &nx, &ny);
fgets (row, 80, inimage);

/* allocate storage */
alloc_matrix (&f2, nx+2, ny+2);

/* read image data */
for (j=1; j<=ny; j++)
 for (i=1; i<=nx; i++)
     f2[i][j] = (float) getc (inimage);
fclose(inimage);


/* ---- read other parameters ---- */

printf("smoothnes weight alpha (>0):      ");
fgets(row, sizeof(row),stdin);  sscanf(row, "%f", &alpha);
printf("number of iterations:             ");
fgets(row,sizeof(row),stdin);  sscanf(row, "%ld", &kmax);
printf("output image (flow magnitude):    ");
fgets(out,sizeof(out), stdin);
printf("\n");


/* ---- initializations ---- */

/* allocate storage for image derivatives fx, fy, fz */
alloc_matrix (&fx, nx+2, ny+2);
alloc_matrix (&fy, nx+2, ny+2);
alloc_matrix (&fz, nx+2, ny+2);

/* calculate image derivatives fx, fy and fz */
dummies (f1, nx, ny);
dummies (f2, nx, ny);
hx = 1.0;
hy = 1.0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     fx[i][j] = (f1[i+1][j] - f1[i-1][j] + f2[i+1][j] - f2[i-1][j]) 
                / (4.0 * hx);
     fy[i][j] = (f1[i][j+1] - f1[i][j-1] + f2[i][j+1] - f2[i][j-1]) 
                / (4.0 * hy);
     fz[i][j] = f2[i][j] - f1[i][j];   /* frame distance 1 assumed */ 
     }
 
/* allocate storage */
alloc_matrix (&u, nx+2, ny+2);
alloc_matrix (&v, nx+2, ny+2);
alloc_matrix (&w, nx+2, ny+2);

/* initialize (u,v) with 0 */
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     u[i][j] = 0.0;
     v[i][j] = 0.0;
     }


/* ---- process image ---- */

for (k=1; k<=kmax; k++)
    {
    /* perform one iteration */
    printf("\n");
    printf("iteration number: %5ld \n", k);
    flow (nx, ny, hx, hy, fx, fy, fz, alpha, u, v);

    /* calculate flow magnitude */
    for (i=1; i<=nx; i++)
     for (j=1; j<=ny; j++)
         w[i][j] = sqrt (u[i][j] * u[i][j] + v[i][j] * v[i][j]);

    /* check minimum, maximum, mean, variance of flow magnitude */
    analyse (w, nx, ny, &min, &max, &mean, &vari);
    printf("maximum:       %8.2f \n", max);
    printf("mean:          %8.2f \n", mean);
    printf("variance:      %8.2f \n", vari);
    } 


/* ---- write output image w (pgm format P5) ---- */

outimage = fopen (out, "wb");
fprintf (outimage, "P5 \n");
fprintf (outimage, "# optic flow magnitude, Horn and Schunck scheme\n");
fprintf (outimage, "# initial image 1:  %s\n", in1);
fprintf (outimage, "# initial image 2:  %s\n", in2);
fprintf (outimage, "# alpha:            %8.4f\n", alpha);
fprintf (outimage, "# iterations:       %8ld\n",  kmax);
fprintf (outimage, "# max:              %8.4f\n", max);
fprintf (outimage, "%ld %ld \n255\n", nx, ny);

/* write image data and close file */
for (j=1; j<=ny; j++)
 for (i=1; i<=nx; i++)
     {
     help = 255.0 * w[i][j] / max;
     if (help < 0.0)
        byte = (unsigned char)(0.0);
     else if (help > 255.0)
        byte = (unsigned char)(255.0);
     else
        byte = (unsigned char)(help);
     fwrite (&byte, sizeof(unsigned char), 1, outimage);
     }
fclose(outimage);

printf("\n");
printf("output image %s successfully written\n\n", out);


/* ---- disallocate storage ---- */

disalloc_matrix (f1, nx+2, ny+2);
disalloc_matrix (f2, nx+2, ny+2);
disalloc_matrix (fx, nx+2, ny+2);
disalloc_matrix (fy, nx+2, ny+2);
disalloc_matrix (fz, nx+2, ny+2);
disalloc_matrix (u,  nx+2, ny+2);
disalloc_matrix (v,  nx+2, ny+2);
disalloc_matrix (w,  nx+2, ny+2);
return(0);
}
