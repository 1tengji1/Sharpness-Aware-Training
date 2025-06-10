# Sharpness-Aware-Training
The open source code for our proposed Physical Neural Network (PNN) training method, Sharpness-Aware Training (SAT).

SAT is a method we proposed based on the well-established method Sharpness-Aware Minimization (SAM) [Paper](https://openreview.net/forum?id=6Tm1mposlrM), and [Code](https://github.com/google-research/sam). 
SAM was originally developed to enhance model generalization in the presence of data distribution shifts. In contrast, SAT aims to generally facilitate PNN's real-world deployment by searching for robust minima to all types of imperfections without re-training.

The key to this work is that it automatically finds the robust minima without requiring any prior knowledge of the physical system.

To maximally increase PNN's robustness, we propose to reformulate weight optimization as control parameter optimization. This change enables the optimization to fully consider the physical relationship between weights and control parameters (In 'System1-MRR Weight Bank-01-Training with different methods.ipynd', we incorporate the physical relation between the Microring weight bank's current and weight into the convolution layer and the linear layer.). Moreover, we further propose to use the established finite difference method to approximate the gradient, thereby enabling SAT to be generally applicable to PNNs even without explicitly known models (In 'System2-D2NN-01-Training and evaluate robustness.ipynb' and 'System2-Free space PNN-01-Training and sweep accuracy', we demonstrate how to use the finite difference method to approximate the gradient).

Our codes are constructed based on many excellent works and well-established techniques.
Novel PNN training works
1. Physical-Aware-Training, [Paper](https://www.nature.com/articles/s41586-021-04223-6), [Code](https://github.com/mcmahon-lab/Physics-Aware-Training)
2. Dual-Adaptive-Training, [Paper](https://www.nature.com/articles/s42256-023-00723-4), [Code](https://github.com/THPCILab/DAT_MPNN)
3. 

# Getting started
The required environment libraries are summarized at the beginning of each code. Please download the libraries first and then run the code.
