#include "mex.h"
#include <math.h>
#include <stdlib.h>

void printf_vec_d(double *arr_double_ind, int n)
 {
     int i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%f, ",arr_double_ind[i]);
         mexEvalString("drawnow");
     }  
     mexPrintf("\n");
 }


void printf_vec_i(int *arr_double_ind, int n)
 {
     int i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%d, ",arr_double_ind[i]);
         mexEvalString("drawnow");
     }  
     mexPrintf("\n");
 }

//Note that in C, index= no_of_rows * col_id + row_id
void printf_mat(double *arr_double_ind, int m, int n)
 {
     int i,j;
     for (i=0;i <m; ++i)
     {
         for (j=0; j<n; j++)
         {
            mexPrintf("%f, ",arr_double_ind[j* m+ i]);
         }
         mexPrintf("\n");
     }    
 }

double sum_r(double *arr, int row, int n_row, int n_col)
{
    int j;
    double sum_=0;
    for (j=0; j<n_col; j++)
    {       
        sum_=sum_+arr[j* n_row + row];
    }
    return sum_;
}

double find_arr(double *arr, int row, int n_row, int n_col, double val)
{
    int j;
    for (j=0; j<n_col; j++)
    {
        if(arr[j * n_row + row]==val)
        {
            return j;
        }
    }
    return -1;
}

//Note that list[count]=count;
double * find_arr_list(double *arr, int row, int n_row, int n_col, double val)
{    
    int j=0;
    int *list;
    int count=0;
    
    for (j=0; j<n_col; j++)
    {
        if(arr[j * n_row + row]==val)
        {
            count=count+1;
        }
    }
    
    list = (int*)mxCalloc((mwSize)(count),(mwSize)sizeof(int));
    
    count=0;
    
    for (j=0; j<n_col; j++)
    {
        if(arr[j * n_row + row]==val)
        {
            list[count]=j;
            count=count+1;
        }
    }
      
    return list;
}

int count_find_arr_list(double *arr, int row, int n_row, int n_col, double val)
{    
    int j=0;
    int count=0;
    
    for (j=0; j<n_col; j++)
    {
        if(arr[j * n_row + row]==val)
        {
            count=count+1;
        }
    }
    
    return count;
}

double * project_row(double *arr, int row, int no_of_rows, int no_of_cols)
{
    double *result= (double*)mxCalloc((mwSize)no_of_cols,(mwSize)sizeof(double));
    
    int j;
    
    for (j=0; j<no_of_cols; j++)
    {
        result[j]=arr[j*no_of_rows+row];
    }
    return result;
}

double * project_col(double *arr, int col, int no_of_rows, int no_of_cols)
{
    double *result= (double*)mxCalloc((mwSize)no_of_rows,(mwSize)sizeof(double));
    
    int i;
    
    for (i=0; i<no_of_rows; i++)
    {
        result[i]=arr[col*no_of_rows+i];
    }
    return result;
}

double * substract_two_vecs(double *arr1, double *arr2, int count)
{
    double *result= (double*)mxCalloc((mwSize)count,(mwSize)sizeof(double));
    
    int i;
    
    for (i=0; i<count; i++)
    {
        result[i]=arr1[i]-arr2[i];
    }
    return result;
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *str, *str1, *str2, *u;
    size_t powerC,C;     
    double *outMatrix1, *outMatrix2;  
    int i, j, k, h, l, t, class_count, dimension, ind;
    double temp1, temp2, temp3;
    double a, b, count; 
    double *path_;
    double *map;
    double *ind_class;
    int *v, *w;
    
    str = mxGetPr(prhs[1]);
    powerC=mxGetM(prhs[1]);
    C = mxGetN(prhs[1]);
    dimension=(int)powerC-1;
    map = mxGetPr(prhs[2]);
    
    ind_class = (double*)mxCalloc((int)C,(mwSize)sizeof(double));  
    memcpy(ind_class, map, (int)C*sizeof(double));
    
    if((int)C>1)
    {
        for (i=1; i<(int)C; i++)
        {
            map[i]=map[i-1]+map[i];
        }
    }
    
    count=map[(int)C-1];
      
    path_ = (double*)mxCalloc((int)count,(mwSize)sizeof(double)); 

    ind = 0;
    
    for (i=0; i<dimension; i++) 
    {
        for (j=0; j<dimension; j++)
        {
            str1=project_row(str, i+1, (int)powerC, (int)C);
            str2=project_row(str, j+1, (int)powerC, (int)C);
                       
            u=substract_two_vecs(str2, str1, (int)C);
                       
            h= count_find_arr_list(u, 0, 1, (int)C, 1);
            
            k= count_find_arr_list(u, 0, 1, (int)C, -1);
                             
           if((h==0)&&(k==1))
           {
               l= find_arr(u, 0, 1, (int)C, -1);
               ind = (int)(map[l] - ind_class[l]);
               path_[ind] = (i+1)*1000000+(j+1);
               ind_class[l] = ind_class[l] -1;
               
           }
           else if((h==0)&&(k==0))
           {
               l= count_find_arr_list(str1, 0, 1, (int)C, 1);
               w=find_arr_list(str1, 0, 1, (int)C, 1);
               for (t=0; t<l; t++)
               {
                   ind = (int)(map[(w[t])] - ind_class[(w[t])]);
                   path_[ind] = (i+1)*1000000+(j+1);
                   ind_class[(w[t])] = ind_class[(w[t])]- 1;
               }
           }
           
//            mxFree(str1);
//            mxFree(str2);
//            mxFree(u);
        }
    } 
    
//     printf_vec_d(path_,(int)count);
    
    plhs[0] = mxCreateDoubleMatrix((mwSize)1, (mwSize)count, mxREAL);
    outMatrix1 = mxGetPr(plhs[0]);
    memcpy(outMatrix1, path_, (int)count*sizeof(double));
//     mxFree(path_);
//     mxFree(ind_class);
    }
    
     
    
