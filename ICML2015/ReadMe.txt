1.--Run SyntheticDataGeneration.m to create a toy dataset
  --Then run MIMLNC_Toy.m to visually see how the algorithm works

2. To demo, MIMLNC.m. We provide two datasets: Letter Frost and HJA in the corresponding dataset folders
+ The instance annotation accuracy, linear feature mode, should be (around with std) 64% for Letter Frost

+ Dataset format: 
  1. X{b} is an d x n_b matrix denotes n_b instances in the bth bag with d is the feature dimension.
  
  2. --y{b} is an n_b x C matrix denotes corresponding ground truth instance labels in the bth bag. C is the number of classes. 
     --The Cth class denotes novel class
     --y{b}(i,j)=1 means that the ith instance has label jth. An instance has only one label.
	 
  3. --Y is an B x (C-1) matrix denotes bag labels of all B bags. Y(b,j)=1 means that the bth bag has label jth. 
     --Note that Y(b,:) does not reveal any information about the existence of Cth class (novel class) instances in the bth bag. 
	 --A bag can have multiple labels.
  
+ If you encounter any issue, please contact Anh T. Pham (phaman@eecs.oregonstate.edu) for help

REFERENCES:
[1] Anh T. Pham, Raviv Raich, Xiaoli Z. Fern, and Jesús Pérez Arriaga, Multi-instance multi-label learning in the presence of novel class instances. ICML 2015