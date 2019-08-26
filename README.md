## Neural Network Graph
 
Library to build, differentiate, and optimize computational graphs. 
Supports differentially private optimization via mixins.

| Folder         | Name                                                    |  
| -----------    | -------------------------------------------------       |  
| Environments   | Classes for wrapping data                               |  
| Neural_Network | Node implementations for building a computational graph |  
| Tests          | Example networks, compatible with pytest                |  

- Optimizers
    * Gradient Descent
    * Momentum
    * Nesterov Accelerated Gradient
    * Adagrad
    * RMSProp
    * Adam
    * Adamax
    * Nadam
    * Quickprop
- Selectable functions
    + Basis / Activation
        * Regression (softplus, tanh, etc.)
        * Classification (use softmax on final layer)
    + Cost (sum squared and cross entropy)
- Network Configuration
    + Construct flexible graphs of vector functions
    + Multiple layers
    + Set number of nodes for each layer
- Optimizer Configuration
    + Batch size (set to one for stochastic descent)
    + Set hyperparameters for each layer, or broadcast one
        * Learning rate
        * Parameters specific to convergence algorithm
    + Multiprocessed graphing


This is a rewrite of a more rigid 'caterpillar' network:
https://github.com/Shoeboxam/Neural_Network
