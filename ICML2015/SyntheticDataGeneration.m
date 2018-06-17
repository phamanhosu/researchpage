function SyntheticDataGeneration
Oreginal_DataGeneration; %Create orginal MIML dataset of 6 classes as follows

Novel_Classes=[1,5];
NovalClass_DataGeneration(Novel_Classes); %Create MIML dataset with novel class by considering some classes (e.g., 1 and 5) are novel

VisualizeBags(Novel_Classes); %Visualize Bags
end


function Oreginal_DataGeneration
%Create orginal MIML dataset of 6 classes
no_og_bags=100;
X=[]; y=[]; 
count=1;
Y=zeros(no_og_bags,6);

for i=1:no_og_bags
     class_arr=ClassArr(6);   
     [X{count},Y(count,:),y{count}]=BagGeneration(class_arr);count=count+1;
end

save('Dataset\Synthetic\allclasses\allclasses.mat','X','Y','y');
end



function NovalClass_DataGeneration(novel_class_arr)
%Create MIML dataset with novel class by considering some classes are novel
file_name_in=strcat('Dataset\Synthetic\allclasses\allclasses.mat');

file_name_out=strcat('Dataset\Synthetic\allclasses\newlabels.mat');

load(file_name_in);

[y, Y, Y_backup]=IgnoreClass(y,novel_class_arr);

[X,Y,y,Y_backup]=RemoveEmptyBag(X,Y,y,Y_backup);

[X,Y,y,Y_backup]=ShuffleBags(X,Y,y,Y_backup);

save(file_name_out,'X','Y','y','Y_backup');
end



function VisualizeBags(Novel_Classes)
%Visualize Bags
load('Dataset\Synthetic\allclasses\newlabels.mat');

close all;
h=figure;

for count=1:6
    if(count==1)
        X_all=cell2mat(X);
        y_all=cell2mat(y'); 
        subplot(2,3, count); 
        PlotResultData(X_all,20, 0, y_all,Novel_Classes);
    else
        subplot(2,3, count);
        PlotBagData(X,Y,20, 0, count);
    end
end
end


%%% ===================================Sub-Functions=================================================
function [X_b,Y_b,y_b]=BagGeneration(class)
X_b=[];
y_b=[];

x1=4;
x2=10;
x3=16;

y1=4;
y3=16;

variance=10;
for i=1:class(1)
    x=randi([0,10],1); x=x-5; x=x/variance;
    x=x+x1;
    y=randi([0,10],1); y=y-5; y=y/variance;
    y=y+y3;
    X_b=[X_b [x;y]];
    y_b=[y_b; [1 0 0 0 0 0]];
end

for i=1:class(2)
    x=randi([0,10],1); x=x-5; x=x/variance;
    x=x+x3;
    y=randi([0,10],1); y=y-5; y=y/variance;
    y=y+y3;
    X_b=[X_b [x;y]];
    y_b=[y_b; [0 1 0 0 0 0]];
end

for i=1:class(3)
    x=randi([0,10],1); x=x-5; x=x/variance;
    x=x+x2;
    y=randi([0,10],1); y=y-5; y=y/variance;
    y=y+y1;
    X_b=[X_b [x;y]];
    y_b=[y_b; [0 0 1 0 0 0]];
end

for i=1:class(4)
    x=randi([0,10],1); x=x-5; x=x/variance;
    x=x+x1;
    y=randi([0,10],1); y=y-5; y=y/variance;
    y=y+y1;
    X_b=[X_b [x;y]];
    y_b=[y_b; [0 0 0 1 0 0]];
end

for i=1:class(5)
    x=randi([0,10],1); x=x-5; x=x/variance;
    x=x+x3;
    y=randi([0,10],1); y=y-5; y=y/variance;
    y=y+y1;
    X_b=[X_b [x;y]];
    y_b=[y_b; [0 0 0 0 1 0]];
end

for i=1:class(6)
    x=randi([0,10],1); x=x-5; x=x/variance;
    x=x+x2;
    y=randi([0,10],1); y=y-5; y=y/variance;
    y=y+y3;
    X_b=[X_b [x;y]];
    y_b=[y_b; [0 0 0 0 0 1]];
end

Y_b=ptoY(y_b);
end



function class_arr=ClassArr(no_of_class)
a = 1/no_of_class*ones(1,no_of_class);
n = 1;
r = drchrnd(a,n);
class_arr=round(r*10);
end



function r = drchrnd(a,n)
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);
end



function Y=ptoY(p)
Y=zeros(1,size(p,2));
for i=1:size(p,1)
    if(sum(p(i,:))>0)
        [~, indclass]=max(p(i,:));
        Y(indclass)=1;
    end
end
end




function PlotResultData(X,maxx, minx, y,Novel_Classes)
PlotData(X,y,maxx, minx,Novel_Classes);
end



function PlotBagData(X,Y,maxx, minx,count)
X=X{count};
Y=Y(count,:);
plot(X(1,:),X(2,:),strcat('.','k'),'MarkerSize',40);
axis([minx maxx minx maxx]);
hold on; 
id=find(Y==1);
id=strrep(num2str(id),'  ',32);
title(strcat('Bag', 32, num2str(count),',', 32,32,'Bag label',32,id));
end




function PlotData(X,p,maxx, minx,novel_classes)
color=['b' 'g' 'r' 'c' 'm' 'y' 'k' 'w' 'b' 'g'];
y=zeros(size(X,2),1);

for i=1:size(X,2)
    [value,label]=max(p(i,:));
    y(i)=label;
end

z=unique(y);

x1=4; x2=10; x3=16; y1=4; y3=16;
class_centers=[x1 x3 x2 x1 x3 x2; y3 y3 y1  y1 y1 y3];

normal_classes=1:1:6; normal_classes(novel_classes)=[];

for i=1:length(normal_classes)
    index=find(y==z(i));
    plot(X(1,index),X(2,index),strcat('.',color(z(i))),'MarkerSize',40);
    axis([minx maxx minx maxx]);
    hold on; 
    
    text(class_centers(1,normal_classes(i)), class_centers(2,normal_classes(i)),strcat('Class',32,num2str(i)));
end

index=find(y==length(z));
plot(X(1,index),X(2,index),strcat('.',color(z(length(z)))),'MarkerSize',40);
axis([minx maxx minx maxx]);
hold on; 

for i=1:length(novel_classes)
    text(class_centers(1,novel_classes(i)), class_centers(2,novel_classes(i)),strcat('Class',32,'novel'));
end

title('Data distribution');
end




function [y_in, Y, Y_backup]=IgnoreClass(y_in, class_id)
allclasses=1:1:size(y_in{1},2);
allclasses(class_id)=[];

Y=zeros(length(y_in),size(y_in{1},2)-length(class_id));
Y_backup=zeros(length(y_in),size(y_in{1},2)-length(class_id)+1);

if(length(class_id)~=0)
    for i=1:length(y_in)
       temp=y_in{i}(:,allclasses);
       Y(i,:)=ptoY(temp);

       new_labels_data=UnionClass(y_in{i}(:,class_id));
       temp=[temp new_labels_data];
       y_in{i}=temp;
    end
else
    for i=1:length(y_in)
        Y(i,:)=ptoY(y_in{i});
    end
end

for i=1:length(y_in)
    Y_backup(i,:)=ptoY(y_in{i});
end
end


function newlabels=UnionClass(new_class)
temp=new_class(:,1);
for i=1:size(new_class,2)
    temp=temp+new_class(:,i);
end
newlabels=temp>0;
end



function [X,Y,y,Y_backup]=RemoveEmptyBag(X,Y,y,Y_backup)
delete=[];
for i=1:length(X)
    if(sum(Y(i,:))==0)
       delete=[delete i]; 
    end
end
X(delete)=[];
Y(delete,:)=[];
y(delete)=[];
Y_backup(delete,:)=[];
end



function [X,Y,y,Y_backup]=ShuffleBags(X,Y,y,Y_backup)
rand_perm_id=randperm(size(Y,1),size(Y,1));
X=X(rand_perm_id);
Y=Y(rand_perm_id,:);
y=y(rand_perm_id);
Y_backup=Y_backup(rand_perm_id,:);
end

