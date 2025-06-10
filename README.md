# Sharpness-Aware-Training
The open source code for our proposed Physical Neural Network (PNN) training method, Sharpness-Aware Training (SAT).
![Sharpness-Aware-Training/fig.jpg](Method figure.jpg)

SAT is a method we proposed based on the well-established method Sharpness-Aware Minimization (SAM) [Paper](https://openreview.net/forum?id=6Tm1mposlrM) and [Code](https://github.com/google-research/sam). 
SAM was originally developed to enhance model generalization in the presence of data distribution shifts. In contrast, SAT aims to generally facilitate PNN's real-world deployment by searching for robust minima to all types of imperfections without re-training.

The key to this work is that it automatically finds the robust minima without requiring any prior knowledge of the physical system. Different from existing training methods that are only valid during the training stage, we pay attention to the system performance post-training under perturbations and imperfections. In fact, SAT aims not to replace the existing methods but to use the gradient information provided by the existing methods to generally improve the PNN's robustness post-training. We envision this as an important step for PNN's real-world applications.

Different from SAM, to maximally increase PNN's robustness, we propose to reformulate weight optimization as control parameter optimization. This change enables the optimization to fully consider the physical relationship between weights and control parameters. (In 'System1-MRR Weight Bank-01-Training with different methods.ipynb', we incorporate the physical relation between the Microring weight bank's current and weight into the convolution layer and the linear layer.) Moreover, we further propose to use the established finite difference method to approximate the gradient, thereby enabling SAT to be generally applicable to PNNs even without explicitly known models. (In 'System2-D2NN-01-Training and evaluate robustness.ipynb' and 'System2-Free space PNN-01-Training and sweep accuracy', we demonstrate how to use the finite difference method to approximate the gradient.)

Our codes are constructed based on many excellent works and well-established techniques.

Novel PNN training works, 
1. Physical-Aware-Training (PAT), [Paper](https://www.nature.com/articles/s41586-021-04223-6), [Code](https://github.com/mcmahon-lab/Physics-Aware-Training)
2. Dual-Adaptive-Training (DAT), [Paper](https://www.nature.com/articles/s42256-023-00723-4), [Code](https://github.com/THPCILab/DAT_MPNN)
3. Noise-Aware-Training (NAT), [Paper1](https://www.nature.com/articles/s41467-022-33259-z), [Code](https://github.com/georgemourgias/noise_aware_cpnn); [Paper2](https://www.science.org/doi/10.1126/sciadv.abm2956); [Paper3](https://ieeexplore.ieee.org/document/9472868); [Paper4](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202200029)

Well-established techniques,
1. Sharpness-Aware Minimization (SAM),  [Paper](https://openreview.net/forum?id=6Tm1mposlrM), [Code1](https://github.com/google-research/sam), [Code2](https://github.com/davda54/sam)
2. Plot loss landscape, [Code](https://github.com/marcellodebernardi/loss-landscapes/tree/master)
3. Evaluate model's robustness, [Code](https://github.com/google/spectral-density)
4. Deep diffraction neural network (D2NN) simulation tools, [Paper](https://www.science.org/doi/10.1126/science.aat8084), [Code](https://github.com/Loli-Eternally/D2NN-with-Pytorch)
5. Mach-Zehnder Interferometer (MZI) mesh simulation tools, [Paper](https://www.nature.com/articles/nphoton.2017.93), [Code](https://github.com/solgaardlab/neurophox)

# Getting started
The required environment libraries are summarized at the beginning of each code. Please download the libraries first and then run the code.

# Cite this work
If you find our code helpful in your research or work, please cite our paper.

```bibtex
@article{xu2024perfecting,
  title={Perfecting Imperfect Physical Neural Networks with Transferable Robustness using Sharpness-Aware Training},
  author={Xu, Tengji and Luo, Zeyu and Liu, Shaojie and Fan, Li and Xiao, Qiarong and Wang, Benshan and Wang, Dongliang and Huang, Chaoran},
  journal={arXiv preprint arXiv:2411.12352},
  year={2024}
}
