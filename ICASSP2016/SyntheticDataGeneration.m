function SyntheticDataGeneration
Oreginal_DataGeneration; %Create orginal SISL dataset of 6 classes

Novel_Classes=[1,6];
Sampling_From_Data(Novel_Classes);
end


function Oreginal_DataGeneration
[X_source,y_source]=XGeneration;    
save('Dataset\Synthetic\allclasses\allclasses.mat','X_source','y_source');
end

function Sampling_From_Data(Novel_Classes)
load('Dataset\Synthetic\allclasses\allclasses.mat');

no_labeled_samples=100; no_unlabeled_samples=1000;

ind=[];
for i=1:length(Novel_Classes)
    temp=find(y_source==Novel_Classes(i));
    ind=[ind; temp];  
end

temp=1:1:length(y_source); temp(ind)=[]; ind=temp;

selected_label_ind=randperm(length(ind),no_labeled_samples);

X_label=X_source(:,ind(selected_label_ind)); X_source(:,ind(selected_label_ind))=[];
y_label=y_source(ind(selected_label_ind)); y_source(ind(selected_label_ind))=[];

selected_unlabel_ind=randperm(length(y_source),no_unlabeled_samples);

X_unlabel=X_source(:,selected_unlabel_ind);
y_unlabel=zeros(1,length(selected_unlabel_ind));
y_unlabel_true=y_source(selected_unlabel_ind);

ind=[];
for i=1:length(Novel_Classes)
    temp=find(y_unlabel_true==Novel_Classes(i));
    ind=[ind; temp]; 
end
y_unlabel_true(ind)=0;

n=length(ind);

labels=unique(y_label);

for label_id=1:length(labels)
    y_label(y_label==labels(label_id))=label_id;
    y_unlabel_true(y_unlabel_true==labels(label_id))=label_id;
end

X=[X_label X_unlabel];
y=[y_label' y_unlabel];
y_truth=[y_label' y_unlabel_true'];

PlotUnlabelData(X,y,100,-100);

[X, y, y_truth]=ConvertDataFormat(X, y, y_truth);

save('Dataset\Synthetic\allclasses\allclasses_sparse.mat','X','y','n','y_truth');
end




function [X_out, y_out, y_truth_out]=ConvertDataFormat(X_in, y_in, y_truth_in)
no_of_class=length(unique(y_in))-1;
for i=1:size(X_in,2)
    X_out{i}=X_in(:,i);
    temp=zeros(1,no_of_class);
    if(y_in(i)>0); temp(y_in(i))=1; end;
    y_out{i}=temp;
    temp=zeros(1,no_of_class);
    if(y_truth_in(i)>0); temp(y_truth_in(i))=1; end;
    y_truth_out{i}=temp;
end

end




function [X_b,y_b]=XGeneration
X0=[100 0 -100 100 0 -100];
Y0=[100 100 100 -100 -100 -100];

X_b=[];
y_b=[];

for i=1:1000
    x=randi([0,50],1); x=x-25; 
    x=x+X0(1);
    y=randi([0,50],1); y=y-25;
    y=y+Y0(1);
    X_b=[X_b [x;y]];
    y_b=[y_b; 1];
end

for i=1:1000
    x=randi([0,50],1); x=x-25; 
    x=x+X0(2);
    y=randi([0,50],1); y=y-25; 
    y=y+Y0(2);
    X_b=[X_b [x;y]];
    y_b=[y_b; 2];
end

for i=1:1000
    x=randi([0,50],1); x=x-25; 
    x=x+X0(3);
    y=randi([0,50],1); y=y-25; 
    y=y+Y0(3);
    X_b=[X_b [x;y]];
    y_b=[y_b; 3];
end

for i=1:1000
    x=randi([0,50],1); x=x-25; 
    x=x+X0(4);
    y=randi([0,50],1); y=y-25; 
    y=y+Y0(4);
    X_b=[X_b [x;y]];
    y_b=[y_b; 4];
end

for i=1:1000
    x=randi([0,50],1); x=x-25; 
    x=x+X0(5);
    y=randi([0,50],1); y=y-25;
    y=y+Y0(5);
    X_b=[X_b [x;y]];
    y_b=[y_b; 5];
end

for i=1:1000
    x=randi([0,50],1); x=x-25; 
    x=x+X0(6);
    y=randi([0,50],1); y=y-25; 
    y=y+Y0(6);
    X_b=[X_b [x;y]];
    y_b=[y_b; 6];
end
end





function PlotUnlabelData(X,y,maxx,minx)
color=['b' 'g' 'r' 'c' 'm' 'y' 'k' 'w' 'b' 'g'];
y=y+1;
z=unique(y);

for i=1:length(z)
    index=find(y==z(i));
    plot(X(1,index),X(2,index),strcat('.',color(z(i))));
    hold on; 
end
grid on;
axis([minx maxx minx maxx]);
end




