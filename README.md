# Sharpness-Aware-Training
The open source code for our proposed Physical Neural Network (PNN) training method, Sharpness-Aware Training (SAT).

SAT is a method we proposed based on the well-established method Sharpness-Aware Minimization (SAM) [Paper](https://openreview.net/forum?id=6Tm1mposlrM), and [Code](https://github.com/google-research/sam). 
SAM was originally developed to enhance model generalization under data distribution shifts. In contrast, SAT aims to generally facilitate PNN's real-world deployment by searching for robust minima to all types of imperfections without re-training.

To maximally increase PNN's robustness, we propose to reformulate weight optimization as control parameter optimization. This change enables the optimization to fully consider the physical relationship between weights and control parameters (In 'System1-MRR Weight Bank-01-Training with different methods.ipynd', we incorporate the physical relation between the Microring weight bank's current and weight into the convolution layer and the linear layer.). Moreover, we further propose to use the established finite difference method to approximate the gradient, thereby enabling SAT to be generally applicable to PNNs even without explicitly known models (In 'System2-D2NN-01-Training and evaluate robustness.ipynb' and 'System2-Free space PNN-01-Training and sweep accuracy', we demonstrate how to use the finite difference method to approximate the gradient).

# Getting started
The required environment libraries are summarized at the beginning of each code. Please download the libraries first and then run the code.
