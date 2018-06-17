function LRSSS_Toy
EMiterations=200;
option{1}=100; %norm 2 regularization

fullfilename='Dataset\Synthetic\allclasses\allclasses_sparse.mat';
load(fullfilename);clear Data;
k=200; g=randn(k,size(X{1},1)); save('g.mat','g');

width_arr=[1e-2,1e-1,1,100];

X_save=X;

close all;
h=figure;

count=1;
for width_id=1:length(width_arr)
    X_old=X_save;
    [X, mean2, std2]=PreprocessingX(X_save,'kernel','RBF',width_arr(width_id)); 
    w0=LoadExistFile(X,y);    

    for no_of_novel_instances=50:150:500
        [w,~,~,~,~,~]=OneCrossValidation(w0,X,y,no_of_novel_instances,EMiterations,option);
        subplot(4,4,count);
        PlotFunctions(X_old,y_truth, y,w,no_of_novel_instances,width_arr(width_id));
        title(strcat('width',32,num2str(width_arr(width_id)),',N',32,num2str(no_of_novel_instances)));
        count=count+1;
        axis off;
    end
end

PlotOriginalData(X_old,y);

saveas(h,'Toy_example','jpg');
end


function [w,p,rllharr,llharr, accuracy, loss]=OneCrossValidation(w0,X_train,y_train, N_train, EMiterations,option)
w=[]; accuracy=0; loss=0;
[X_train,y_train]=UnionUnlabeledData(X_train,y_train);
[w,p,rllharr,llharr]=ExpectMaximizationMIMIL(w0,X_train,y_train,N_train, EMiterations,option);
end


function [X_out, y_out]=UnionUnlabeledData(X_in,y_in)
for i=1:length(y_in)
    if(sum(y_in{i})==0)
        X_unlabel=X_in(i:length(X_in)); X_unlabel=cell2mat(X_unlabel);
        y_unlabel=y_in{i}; 
        break;
    end
end
X_label=X_in(1:i-1); X_label=cell2mat(X_label);
y_label=y_in(1:i-1); y_label=cell2mat(y_label');
X_out{1}=X_label; X_out{2}= X_unlabel;
y_out{1}=y_label; y_out{2}= y_unlabel;
end


function [w,p,rllharr,llharr]=ExpectMaximizationMIMIL(w,X,y,N,EMiterations,option)
count=1; step=1;
rllharr=zeros(1,EMiterations); llharr=zeros(1,EMiterations);

while(count<=EMiterations)
for i=1:length(X)
    if(i==1)
        p{i}=ExpectationStep_Label(y{i});
    else
        p{i}=ExpectationStep_UnLabel(X{i},N,w);
    end
end
factor_coeff=option{1};
% rllharr(count)=RealLoglikelihood(X,y,N,w,factor_coeff);

step=10*step;
[w, step, ~]=MaximizationStep(X,size(w,2),w,p,step,factor_coeff);
count=count+1
end

end


%%==============================================================================================
function p=ExpectationStep_Label(y)
p=[zeros(size(y,1),1) y];
end

function p=ExpectationStep_UnLabel(X,N,w)
p=zeros(size(X,2),size(w,2));
prior=PriorOneBag(X,w);
lastindex=size(X,2);

other_classes=1:1:size(w,2); other_classes(1)=[];

pdynamic_full_table=DynamicProgramming_mex(lastindex,prior);

for i=1:size(p,1) 
    lastindex=size(X,2);
    prior(:,[i,lastindex])=prior(:,[lastindex,i]);
    pdynamic=ForwardAndSubstitution(pdynamic_full_table,sum(prior(other_classes,lastindex)),prior(1,lastindex));        
    p(i,:)=PosteriorProbability(X,N,prior,pdynamic);
    prior(:,[i,lastindex])=prior(:,[lastindex,i]);
end
end


function pdynamic=ForwardAndSubstitution(pdynamic_full_table, p_other,p_c)
pdynamic=zeros(length(pdynamic_full_table),1);
pdynamic(1)=pdynamic_full_table(1)/p_other;
for i=2:length(pdynamic_full_table)
    pdynamic(i)=(pdynamic_full_table(i)-pdynamic(i-1)*p_c)/p_other;
    if(pdynamic(i)<0)
        pdynamic(i)=0;
    end
end
end



function pout=PosteriorProbability(X,N,prior,pdynamic)
l=1;
currentsample=size(X,2);
possiblelabel=1:1:size(prior,1);
p=zeros(size(prior,1),1);

p_case1=pdynamic(N+1);
p_case2=pdynamic(N);

for j=1:length(possiblelabel) 
    if(possiblelabel(j)~=l) 
      temp2=prior(possiblelabel(j),currentsample);
      sum_=p_case1*temp2;          
    else 
      temp2=prior(l,currentsample);  
      sum_=p_case2*temp2;  
    end
    p(possiblelabel(j))=sum_;
end
pout=p/sum(p);
end




%%==============================================================================================================
function [w_new, step, enough]=MaximizationStep(X,C,w,p,step_init,factor_coeff)
grad=Gradient(X,C,w,p,factor_coeff);
[w_new,step, enough]=BackTracking(w,0.5,0.7,grad,p,X,C,step_init,factor_coeff);
end




function grad=Gradient(X,C,w,p,factor_coeff)
grad=zeros(size(w,1),size(w,2));
for i=1:C  
    for b=1:length(X)  %%---------------------------------------------------------------------------------      
      prior=PriorOneBag(X{b},w);prior=prior';
      sum1{b}=X{b}*p{b}(:,i);
      sum2{b}=X{b}*prior(:,i);
      if(b==length(X))
          sum1{b}=sum1{b};
          sum2{b}=sum2{b};
      end
    end
    sum1arr=cell2mat(sum1); sum2arr=cell2mat(sum2);
    sum1x=sum(sum1arr,2); sum2x=sum(sum2arr,2);
grad(:,i)= sum1x-sum2x;  
end

grad=grad-2*factor_coeff*w;
end



function p=ExpSubstractOneBag(W,X)
wx_in=W'*X;
[wx_max,wx_out]=SubstractWX(wx_in);
p=exp(wx_out);
end



function p=PriorOneBag(X,W)
pro=ExpSubstractOneBag(W,X);
sumpro=sum(pro);
one_=ones(1,length(sumpro));
invsumpro=one_./sumpro;
p=pro*diag(invsumpro);
end

function [w_new, step, enough]=BackTracking(w,alpha,beta,f1,p,X,C,step_init,factor_coeff)
stop=0;
enough=0;
f=LoglikelihoodRegularization(X,p,w,factor_coeff);
w0=w;
step=step_init;
while(stop==0)
    w_step=w0+step*f1;
    w_copy=w_step;
    f_w_step=LoglikelihoodRegularization(X,p,w_copy,factor_coeff);
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


function llh_regularization=LoglikelihoodRegularization(X,p,w,factor_coeff)
llh_regularization=Loglikelihood(p,X,w,factor_coeff);
end


function llh=Loglikelihood(p,X,w,factor_coeff)
llh=zeros(length(X),1);
for b=1:length(X)           %%--------------------------------------------------------------------------------- 
    llh(b)=LoglikelihoodOneBag(p{b},X{b},w);
    if(b==length(X))
       llh(b)=llh(b);
    end
end
llh=sum(llh);
llh=llh-factor_coeff*trace(w'*w);
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





function [max_out,wx_out]=SubstractWX(wx_in)
max_out=max(wx_in);
max_out_matrix=ones(size(wx_in,1),1)*max_out;
wx_out=wx_in-max_out_matrix;
end




function [X_out, mean2, std2]=PreprocessingX(X,kernel,kernel_method,scale)
w = warning ('on','all');rmpath('folderthatisnotonpath');warning(w);id = w.identifier;warning('off',id);rmpath('folderthatisnotonpath');
mean2=0;
std2=0;
if(strcmp(kernel,'kernel')==1)
    X_out=Kernelize(X,kernel_method,scale);
end
if(strcmp(kernel,'feature')==1)
    
end
end



function X_out=AddOne(X)
for i=1:length(X)
    X_out{i}=[X{i}; ones(1,size(X{i},2))];
end
end


function X_out=Kernelize(X,kernel_method,scale)
load('g.mat');
g=g*scale;
for i=1:length(X)
    X_out{i}=[cos(g*X{i}); sin(g*X{i})];
end
end

function [X_out, mean2, std2]=DataNormalize(X)

firstmm=zeros(size(X{1},1),length(X));
secondmm=zeros(size(X{1},1),length(X));
n=0;

for i=1:length(X)
firstmm(:,i)=sum(X{i},2);
secondmm(:,i)=sum(X{i}.*X{i},2);
n=n+size(X{i},2);
end

mean2=sum(firstmm,2)/n;

secondmm2=sum(secondmm,2)/n;

variance2=secondmm2-mean2.*mean2;
std2=sqrt(variance2*n/(n-1));
ind=find(std2==0);std2(ind)=abs(mean2(ind))+1;

for i=1:length(X)
    for j=1:size(X{i},2)
        temp=X{i}(:,j)-mean2;
        X{i}(:,j)=temp./std2;
    end
end

X_out=X;
end

function w=LoadExistFile(X,y)
w0=Initializew(X,y{1},{'rand'}); save('w0.mat','w0');
w=w0;
end

function w=Initializew(X,Y,option)
w=randi(10,size(X{1},1),size(Y,2)+1);
w=normalize(w);
end

function w_new=normalize(w)
sum_column=ones(1,size(w,2))./sum(w);
w_new=w*diag(sum_column);
end


function PlotFunctions(X_old,y_truth,y,w,no_of_instances,width)
PlotResultBoundary(X_old,w,no_of_instances,width);
end



function PlotResultBoundary(X,w,no_of_instances,width)
X=cell2mat(X);
minx=min(X(1,:));
maxx=max(X(1,:));

X=randi([minx*100,maxx*100],2,10000);
X=X/100; X_bound_data=X;

load('g.mat'); g=g*width;

X=[cos(g*X); sin(g*X)];

p=PriorOneBag(X,w); p=p';
PlotData(X_bound_data,p,maxx, minx);
end

function PlotData(X,p,maxx, minx)
color=['b' 'g' 'r' 'c' 'm' 'y' 'k' 'w' 'b' 'g'];
y=zeros(size(X,2),1);

for i=1:size(X,2)
    [value,label]=max(p(i,:));
    y(i)=label;
end

z=unique(y);

for i=1:length(z)
    index=find(y==z(i));
    plot(X(1,index),X(2,index),strcat('.',color(z(i))),'MarkerSize',40);
    axis([minx maxx minx maxx]);
    hold on; 
end
end




function PlotOriginalData(X,p)
color=['k' 'g' 'r' 'c' 'm' 'y' 'b' 'w' 'b' 'g'];
X=cell2mat(X);
p=cell2mat(p');
y=zeros(size(X,2),1);
minx=min(X(1,:));
maxx=max(X(1,:));

for i=1:size(X,2)
    if(sum(p(i,:))~=0)
        [value,label]=max(p(i,:));
        y(i)=label;
    else
        y(i)=0;
    end
end

y=y+1;
z=unique(y); 

h=figure; 
for i=1:length(z)
    index=find(y==z(i));
    point_size=50;
    if(i==1);point_size=8;end;
    plot(X(1,index),X(2,index),strcat('.',color(z(i))),'MarkerSize',point_size);
    axis([minx maxx minx maxx]);
%     axis off;
    hold on; 
end
end


