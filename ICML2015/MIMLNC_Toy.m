function MIMLNC_Toy
% Note: Run SyntheticDataGeneration.m first to create Toy dataset

% Run MIMLNC with different values of lambda and RBF kernel width
reg_arr=[1e-1];
width_arr=[1e-2,1e-1,1,100];

close all; h=figure;

count=1;
for i=1:length(reg_arr)
    for j=1:length(width_arr)
        subplot(2,2,count);
        MIML_NC_Core(reg_arr(i), width_arr(j));
        title(strcat('lambda',32,num2str(reg_arr(i)),',width',32,num2str(width_arr(j))));
        count=count+1;
    end
end

saveas(h,'Toy_example','jpg');
end


function MIML_NC_Core(gamma,width)
fullfilename=strcat('Dataset\Synthetic\allclasses\newlabels.mat');

EMiterations=200;

load(fullfilename);clear Data;

kernel='kernel'; kerneltype='RBF';

minx=0; maxx=20; [X, ~]=AddBoundBag(X,minx,maxx,100); X_bound_data=X{length(X)}; X_data=X(1:length(X)-1); 

k=200; g=randn(k,size(X{1},1)); save('g.mat','g');

[K,X]=PreprocessingX(X,kernel,kerneltype,width); 

X_bound_kernel=X{length(X)}; X_data_kernel=X(1:length(X)-1);
X=X_data_kernel; 

w0=LoadExistFile(X,Y);

[w,~,~,~]=ExpectMaximizationMIMIL(w0,K,X,Y,EMiterations,gamma);

PlotResultBoundary(X_bound_kernel,X_bound_data,maxx, minx, w);
end




function [X, X_bound]=AddBoundBag(X,minx,maxx,resolution)
u = linspace(minx,maxx, resolution);
v = linspace(minx, maxx, resolution);
X_bound=[];
for i = 1:length(u)
    for j = 1:length(v)
        X_bound=[X_bound [u(i) v(j)]'];
    end
end
X{length(X)+1}=X_bound;
end




function PlotResultBoundary(X_bound,X_bound_data,maxx, minx, w)
p=PriorOneBag(X_bound,w); p=p';
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
    plot(X(1,index),X(2,index),strcat('.',color(z(i))),'MarkerSize',20);
    axis([minx maxx minx maxx]);
    hold on; 
end
end





%% Utility section
function w_new=normalize(w)
sum_column=ones(1,size(w,2))./sum(w);
w_new=w*diag(sum_column);
end
 


function w=Initializew(X,Y)
%One class for new label
w=randi(1000,size(X{1},1),size(Y,2));
end
 



function X_out=DataNormalize(X)
firstmm=zeros(size(X{1},1),length(X));
secondmm=zeros(size(X{1},1),length(X));
n=0;
 
for i=1:length(X)
    if(size(X{i},2)>0)
        firstmm(:,i)=sum(X{i},2);
        secondmm(:,i)=sum(X{i}.*X{i},2);
        n=n+size(X{i},2);
    end
end
 
mean2=sum(firstmm,2)/n;
 
secondmm2=sum(secondmm,2)/n;
 
variance2=secondmm2-mean2.*mean2;
std2=sqrt(variance2*n/(n-1));
ind=find(std2==0);std2(ind)=abs(mean2(ind))+1;
 
for i=1:length(X)
    if(size(X{i},2)>0)
        for j=1:size(X{i},2)
            temp=X{i}(:,j)-mean2;
            X{i}(:,j)=temp./std2;
        end
    end
end
 
X_out=X;
end


 
function X_out=AddOne(X)
for i=1:length(X)
    if(size(X{i},2)>0)
        X_out{i}=[X{i}; ones(1,size(X{i},2))];
    end
end
end



function [K,X_out]=Kernelize(X,kerneltype,width)
load('g.mat'); g=g*width;

for i=1:length(X)
    X_out{i}=[cos(g*X{i}); sin(g*X{i})];
end

K=cell2mat(X_out(1:1:(length(X_out))));
end


 
 
function [K,X_out]=PreprocessingX(X,kernel,kernel_method,scale)
w = warning ('on','all');
rmpath('folderthatisnotonpath');
warning(w);
id = w.identifier;
warning('off',id);
rmpath('folderthatisnotonpath');

if(strcmp(kernel,'kernel')==1)
    [K,X_out]=Kernelize(X,kernel_method,scale);
end
if(strcmp(kernel,'feature')==1)
    X_out=AddOne(X);
    K=zeros(size(X_out{1},1));
end
end
 


function w=LoadExistFile(X,Y)
w0=Initializew(X,Y); w0=[w0 randi(1000,size(w0,1),1)]; w0=w0/1000;
w=w0;
w=normalize(w);
end
 


