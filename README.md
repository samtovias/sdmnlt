# sdmnlt
Learning smooth dendrite morphological neurons for pattern classification using linkage trees and evolutionary-based hyperparameter tuning

 First run "run_first.m" file.
 
 This material contains the necessary functions for
 the reproducibility of the work presented in the paper 
 "Learning smooth dendrite morphological neurons for pattern classification
 ... using linkage trees and evolutionary-based hyperparameter tuning" 
 divided in seven subfolders: 
   
   C-functions/Source-C-Codes 
   -----------
       Shuffle.c - Random permutation of array elements
                   NOTE: For C code function 'Shuffle.c' 
                         could be necessary to re-compile 
                         from the source code using the 
                         mex function.  
       
   datasets
   --------------
       imbalanced - Imbalanced datasets 
       large      - Large datasets
       noisy      - Noisy datasets
       real       - Real datasets
       synthetic  - Synthetic datasets
       
   normalization 
   --------------
       minmaxnorm - Min-max normalization for n x d format data (instances x dimensions)
    
   performance_metrics 
   -------------------
       mulclassperf - Performance metrics    
    
   resampling 
   --------------
       split_data - Split the data into training and validation sets    
   
   results
   --------------
       sdmnlt - Results of the proposed SDMN-LT model
       
   sdmnlt 
   --------------
       crossover        - Crossover operator for micro-GA 
       decode           - Decode binary solutions  
       meanhamming      - Mean Hamming distance for check convergence in micro-GA    
       objfun           - Objective function of micro-GA 
       sdmnlt_mga       - Smooth dendrite morphological neurons using linkage trees 
       sdmnlt_plot      - Plot dendrites and decision regions of the SDMN-LT model
       sdmnlt_predict   - Classify data with a smooth dendrite morphological neuron model 
       sdmnlt_train     - Train a smooth dendrite morphological neuron model with stochastic gradient descent 
       sdmnlt_train_mga - Train a SDMN model with SGD in the objective function of the micro-GA 
       selection        - Selection operator for micro-GA 
       setparams        - Setup the parameters of the proposed algorithm 
