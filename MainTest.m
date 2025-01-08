clear all
clc
addpath([pwd, '/funs']);
addpath([pwd, '/datasets']);
%% load Dataset
datasetName = 'MSRC';
load([datasetName, '.mat']);
gt = Y;
num_C = length(unique(gt));               % 聚类数  
num_V = length(X);                              % 视图个数  
num_N = size(X{1},1);                           % 样本点数  
for v = 1:num_V
    X{v}=X{v}';
    a = max(X{v}(:));
    X{v} = double(X{v}./a);  
end

beta=8700;
p=0.5;
mm=10;
[M, alpha] = myAlgorithm(X,num_N,num_C,num_V, p, beta,mm);
M_sum = M{1} / alpha(1);
for v = 2:num_V
    M_sum = M_sum + M{v} / alpha(v);
end
alpha_sum = sum(1 ./ alpha);
M_final = M_sum / alpha_sum;
[~, Y_pre1] = max(M_final, [], 2); 
my_result1 = ClusteringMeasure1(gt, Y_pre1)

