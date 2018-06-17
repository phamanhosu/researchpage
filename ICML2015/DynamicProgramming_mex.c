#include "mex.h"
#include <math.h>
#include <stdlib.h>

void printf_vec(double *arr_double_ind, int n)
 {
     int i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%f\n",arr_double_ind[i]);
         mexEvalString("drawnow");
     }    
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

int find_arr(double *arr, int row, int n_row, int n_col, double val)
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

int * find_arr_list(double *arr, int row, int n_row, int n_col, int list_count, double val)
{
    int *list= (int*)mxCalloc((mwSize)list_count,(mwSize)sizeof(int));
    
    int j=0;
    
    int count=0;
    
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

//Note that in C, index= no_of_rows * col_id + row_id
double *normalize_mat(double *arr_double_ind, int m, int n, int col_id)
 {
    double sum_=0;
    int i;
    for (i=0; i<m; i++)
    {       
        sum_=sum_+arr_double_ind[m* col_id + i];
    }
    for (i=0; i<m; i++)
    {       
        arr_double_ind[m* col_id + i]=arr_double_ind[m* col_id + i]/sum_;
    }
    return arr_double_ind;
    
 }


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *allsample;
    double *prior;
    double *str;
    double *shift_table;
    double *p;  
    double *arr;
    double *result;
    int *class_list;
    size_t powerC;     
    size_t C;
    double *outMatrix;  
    int i, j,k, h, class_count;
    double l;
    double temp1, temp2, temp3;
    double a, b, c;
    
    allsample = mxGetPr(prhs[0]);
    prior = mxGetPr(prhs[1]);
    str = mxGetPr(prhs[2]);
    shift_table = mxGetPr(prhs[3]);
    
    powerC = mxGetM(prhs[2]);
    C = mxGetN(prhs[2]);
    
    arr = (double*)mxCalloc((mwSize)(int)((int)powerC*(*allsample)),(mwSize)sizeof(double));
    
    for (i=0; i <(int)powerC; i++)
    {
        for (j=0; j<(int)(*allsample);j++)
        {
            arr[(int)(j*((int)powerC)+i)]=0;
        }
    }
    
    for (i=0; i <(int)powerC; i++)
    {
        a=sum_r(str,i,(int)powerC,(int)C);
        if((int)a==1)
        {
            j=find_arr(str,i,(int)powerC,(int)C,1);
            arr[i]=prior[j];
        }
    } 
    
    class_count=sum_r(str,powerC-1,powerC,C);

    class_list=find_arr_list(str, powerC-1, powerC, C, class_count, 1);
// 
//     for curr=2:allsample
//     for i=1:size(str,1)
//       sum_=0;
//       for j=1:length(possiblelabel)
//           strshift=shift_table(i,j);
//           if(strshift~=-1)
//           temp1=arr(i,curr-1)+arr(strshift,curr-1);  
//           temp2=prior(possiblelabel(j),curr);
//           sum_=sum_+temp1*temp2;
//           end 
//       end
//       arr(i,curr)=sum_;
//     end
//     end
    
//     printf_mat(arr, (int)powerC, (*allsample));
//     printf_mat(shift_table, (int)powerC, class_count);
            
    for (k=1; k<(*allsample); k++)
    {
        for (i=0; i<(int)powerC; i++)
        {
            temp3=0;
            for (j=0; j<class_count; j++)
            {
                l=shift_table[j*(int)powerC+i];
                if(l!=-1)
                {
                    l=l-1;
//                     mexPrintf("%d, %d, %d\n", k, i, j);
                    temp1=arr[(k-1)*(int)powerC+ i] + arr[(k-1)*(int)powerC+ (int)l];
                    temp2=prior[k*(int)C+(class_list[j])];
//                     mexPrintf("%f, %f, %f\n", arr[(k-1)*(int)powerC+ i], arr[(k-1)*(int)powerC+ (int)l], temp2);
                    temp3=temp3+temp1*temp2;
                }
                
            } 
            arr[k*(int)powerC+i]=temp3;
        }
        arr=normalize_mat(arr,(int)powerC,(int)(*allsample),k);
    }
    
//     printf_mat(arr, (int)powerC, (*allsample));
    
    result=project_col(arr,(*allsample)-1,(int)powerC, (*allsample));
//     printf_vec(result,(int)powerC);
    
    
    plhs[0] = mxCreateDoubleMatrix(powerC,(mwSize)1,mxREAL);
    outMatrix = mxGetPr(plhs[0]);
//       
//     outMatrix = project_col(arr,(*allsample)-1,(int)powerC, (*allsample));
    memcpy(outMatrix,result, (int)powerC*sizeof(double));
//     printf_vec(outMatrix,(int)powerC);
    }
    
     
    
