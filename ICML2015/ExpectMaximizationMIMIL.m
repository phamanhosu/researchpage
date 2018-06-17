function [w,p,rllharr,llharr]=ExpectMaximizationMIMIL(w,K,X,Y,EMiterations,gamma)
count=1;
step=1;
rllharr=zeros(1,EMiterations);
llharr=zeros(1,EMiterations);

for i=1:length(X)
    path_in{i}=[]; map_in{i}=[];
end
 
for i=1:length(X)
    shift_table_in{i}=[];
end
 
while(count<=EMiterations)
for i=1:length(X)
    [p{i}, shift_table_in{i}, path_in{i}, map_in{i}]=ExpectationStep(X{i},Y(i,:),w,shift_table_in{i},path_in{i},map_in{i});
end
rllharr(count)=RealLikelihood(X,Y,w);
llharr(count)=LoglikelihoodRegularization(K,X,p,w,gamma);
step=10*step;
[w, step, ~]=MaximizationStep(K,X,Y,w,p,gamma,step);
count=count+1
end

end
 
 
function [p,shift_table_out,path_out,map_out]=ExpectationStep(X,Y,w,shift_table_in, path_in, map_in)

classes_in_bag=sum(Y);

if(classes_in_bag>0)

no_of_class=size(Y,2)+1;
p=zeros(size(X,2),no_of_class);
str=PowerSet([Y 1]);

prior=PriorOneBag(X,w);
lastindex=size(X,2);
 
if(isempty(shift_table_in))
    shift_table_out=BuildShiftTable(str);
else
    shift_table_out=shift_table_in;
end
if(isempty(path_in))
    map_out=RememberMap_mex(no_of_class,str);
    path_out=RememberPath_mex(no_of_class,str,map_out);
else
    path_out=path_in; 
    map_out=map_in;
end
 
pdynamic_complte_table=DynamicProgramming_mex(lastindex,prior,str,shift_table_out);
 
Aempty=zeros(size(str,1)-1);Aempty=sparse(Aempty);
class_arr=find(str(size(str,1),:)==1);
for i=1:size(p,1)   
    A=ConstructAMatrix_mex(Aempty,prior(:,i),class_arr',path_out', map_out);A=sparse(A);
    pdynamic=A\pdynamic_complte_table(2:length(pdynamic_complte_table));pdynamic=[0;pdynamic];
    ind=find(pdynamic<0);pdynamic(ind)=0;
    if(lastindex>1)  
        p(i,:)=PosteriorProbability(i,Y,str,prior,pdynamic);
    else
        labels=find(Y(1,:)==1);
        sum_=sum(prior(labels,i));
        for j=1:size(p,2) 
            if(length(find(labels==j))>0)
            p(i,j)=prior(j,i)/sum_;
            end
        end
    end
end

else
    no_of_class=size(Y,2)+1;
    p=zeros(size(X,2),no_of_class);
    str=PowerSet([Y 1]);
    if(isempty(shift_table_in))
        shift_table_out=BuildShiftTable(str);
    else
        shift_table_out=shift_table_in;
    end
    if(isempty(path_in))
        map_out=RememberMap_mex(no_of_class,str);
        path_out=RememberPath_mex(no_of_class,str,map_out);
    else
        path_out=path_in; 
        map_out=map_in;
    end
   p(:,no_of_class)=p(:,no_of_class)+1;
end
end
 
 

function [w_new, step, enough]=MaximizationStep(K,X,Y,w,p,gamma,step_init)
grad=Gradient(K,X,Y,w,p,gamma);
[w_new,step, enough]=BackTracking(K,w,0.5,0.7,grad,p,X,gamma,step_init);
end
 

function grad=Gradient(K,X,Y,w,p,gamma)
grad=zeros(size(w,1),size(w,2));
for i=1:size(Y,2)+1  
    for b=1:length(X)        
      prior=PriorOneBag(X{b},w);prior=prior';
      sum1{b}=X{b}*p{b}(:,i);
      sum2{b}=X{b}*prior(:,i);
    end
    sum1arr=cell2mat(sum1); sum2arr=cell2mat(sum2);
    sum1x=sum(sum1arr,2); sum2x=sum(sum2arr,2);
grad(:,i)= sum1x-sum2x;  
end
 
grad=grad-gamma*2*w; 
end

 
 
function llh_regularization=LoglikelihoodRegularization(K,X,p,w,gamma)
llh_regularization=Loglikelihood(p,X,w);
llh_norm=gamma*trace(w'*w);
llh_regularization=llh_regularization-llh_norm;
end
 

 
function [w_new, step, enough]=BackTracking(K,w,alpha,beta,f1,p,X,gamma,step_init)
stop=0;
enough=0;
f=LoglikelihoodRegularization(K,X,p,w,gamma);
w0=w;
step=step_init;
while(stop==0)
    w_step=w0+step*f1;
       
    w_copy=w_step;
    
    f_w_step=LoglikelihoodRegularization(K,X,p,w_copy,gamma);
    if(f_w_step>f+alpha*step*trace(f1'*f1))     
            stop=1;
    end
    step=step*beta;
    if ((step<1e-12)&&(f_w_step>=f))
        stop=1;
        enough=1;
    end
    if (step<1e-20)
        stop=1;
        enough=1;
    end
end
w_new=w_step;
end



function pout=PosteriorProbability(i,Y,str,prior,pdynamic)

currentsample=i;

possiblelabel=find(Y==1); 

% New label
possiblelabel=[possiblelabel size(str,2)];

p=zeros(size(Y,2)+1,1);

for j=1:length(possiblelabel)
    Y_full=[Y 1];
%     Y_wihout_new_label=[Y 0];
    
    strshift1=Shift(str,possiblelabel(j),Y_full);
    sum_=0;
    if(strshift1~=-1)
    temp1=pdynamic(size(str,1))+pdynamic(strshift1);  
    temp2=prior(possiblelabel(j),currentsample);
    sum_=sum_+temp1*temp2;
    end
    
    if(j~=length(possiblelabel))
        strshift2=strshift1-1;
        if(strshift2~=-1)
        current_str_ind=size(str,1)-1;
        temp1=pdynamic(current_str_ind)+pdynamic(strshift2);  
        temp2=prior(possiblelabel(j),currentsample);
        sum_=sum_+temp1*temp2;
        end
    end
    
    p(possiblelabel(j))=sum_;
end
pout=p/sum(p);
end
 


function shift_table=BuildShiftTable(str)
possiblelabel=find(str(size(str,1),:)==1);
shift_table=zeros(size(str,1),length(possiblelabel));
for j=1:length(possiblelabel)
    for i=1:size(str,1)
        shift_table(i,j)=Shift(str,possiblelabel(j), str(i,:));
    end
end
end
 


function strshift=Shift(str,index, currentstr)
tempstr=(str(size(str,1),:)==1);
if(currentstr(index)==1)
currentstr(index)=0;
currentstr=currentstr(tempstr);
d=bi2de_new(flip(currentstr));
strshift=d+1;
else
   strshift=-1;  
end
end



function d=bi2de_new(b)
max_length = 1024;
pow2vector = 2.^(0:1:(size(b,2)-1));
size_B = min(max_length,size(b,2));
d = b(:,1:size_B)*pow2vector(:,1:size_B).';
end
 

function youtput=PowerSet(yindex)
positivelabelindex=find(yindex==1);
y=zeros(length(positivelabelindex),1);
str=Generate(y,2)-1;
youtput=zeros(size(str,1),size(yindex,2));
for i=1:size(str,1)
    youtput(i,positivelabelindex)=str(i,:);
end
end


function str=Generate(y,c)
if (length(y)==1)
    str=zeros(1,c)';
    for i=1:c
        str(i)=i;
    end
else
   half=floor(length(y)/2);
   str1=Generate(y(1:half),c);
   str2=Generate(y(half+1:length(y)),c);
   I1=zeros(size(str2,1),size(str1,1)); M1=[];
   for j=1:size(str1,1)
       I1(:,j)=1;
       M1=cat(1,M1,I1);
       I1(:,j)=0;
   end
   Mstr1=M1*str1;
   I2=eye(size(str2,1)); M2=[];
   for i=1:size(str1,1);
       M2=cat(1,M2,I2);
   end
   Mstr2=M2*str2;
   str=cat(2,Mstr1,Mstr2);
end
end

 
function llh=Loglikelihood(p,X,w)
llh=zeros(length(X),1);
for b=1:length(X)
    llh(b)=LoglikelihoodOneBag(p{b},X{b},w);
end
llh=sum(llh);
end
 

function llh=LoglikelihoodOneBag(p,X,w)
wX=w'*X;
[wX_max, wX_substract]=SubstractWX(wX);
sum1=sum(sum(p.*(wX)'));
exp_wX=exp(wX_substract);
sum_wX_c=sum(exp_wX);
sum2=sum(log(sum_wX_c));
sum2=sum2+sum(wX_max);
llh=sum1-sum2;
end



function rllh=RealLikelihood(X,Y,w)
rllh=zeros(length(X),1);
for b=1:length(X)
    rllh(b)=RealLikelihoodOneBag(X{b},Y(b,:),w);
end
rllh=sum(rllh);
end


function rllh=RealLikelihoodOneBag(X,Y,w)
str=PowerSet([Y 1]);
prior=PriorOneBag(X,w);
shift_table_out=BuildShiftTable(str);
p=DynamicProgramming_rllh_mex(size(X,2),prior,str,shift_table_out);
sum=p(size(str,1))+p(size(str,1)-1);
rllh=log(sum);
end