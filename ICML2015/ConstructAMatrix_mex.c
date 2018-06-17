#include "mex.h"
#include <math.h>
#include <stdlib.h>

void printf_new (double *arr_double_ind, mwSize n)
 {
     mwSize i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%f\n",arr_double_ind[i]);
         mexEvalString("drawnow");
     }    
 }

void printf_new1 (double *arr_double_ind, int n)
 {
     int i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%f\n",arr_double_ind[i]);
         mexEvalString("drawnow");
     }    
 }

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *A;
    double *prior;
    double *class_arr;
    double *path_in;
    double *map;
    double *arr_double_ind;
    mwSize class_id;
    int index1, index2, index3, row, col;
    int size_arr;
    int n;
    int value_;
    int start_ind;
    double odata[2] = {2, 6}; double *podata;
    double cdata[2] = {1, 4}; double *pcdata;
    
    size_t powerC;     
    size_t C;
    double *outMatrix;   
    
    A = mxGetPr(prhs[0]);
    prior = mxGetPr(prhs[1]);
    class_arr = mxGetPr(prhs[2]);
    path_in = mxGetPr(prhs[3]);
    map = mxGetPr(prhs[4]);
    
    powerC = mxGetN(prhs[0]);
    C= mxGetM(prhs[2]);
    
    plhs[0] = mxCreateDoubleMatrix((mwSize)powerC,(mwSize)powerC,mxREAL);
    outMatrix = mxGetPr(plhs[0]);
        
    for( class_id= 0; class_id <C; class_id++)
    {
       index1=(int)class_arr[class_id];
       if(index1==1)
       {              
           size_arr=(int)map[index1-1];
//            mxFree(arr_double_ind);
           arr_double_ind=(double*)mxCalloc((mwSize)size_arr,(mwSize)sizeof(double));
           memcpy(arr_double_ind,path_in, size_arr*sizeof(double));    
       }
       else
       {           
           start_ind=(int)map[index1-2]+1;
           size_arr=(int)(map[index1-1]-map[index1-2]);
//            mxFree(arr_double_ind);
           arr_double_ind=(double*)mxCalloc((mwSize)size_arr,(mwSize)sizeof(double));
           memcpy(arr_double_ind,path_in+start_ind-1, size_arr*sizeof(double)); 
       }
       for ( index2= 0; index2< size_arr; ++index2)
       {
           value_= arr_double_ind[index2];      
           col=(int)value_/1000000;
           row=value_%1000000;   
           outMatrix[(row-1)* powerC+ col-1]= outMatrix[(row-1)* powerC+ col-1] + prior[index1-1];
       } 
    }
    
}


