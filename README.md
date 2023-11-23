# Analytics Journal

Collection of different topics in data science & computational methods and their implementation examples.

These serves as my notes towards general analytics applications, to be used as quick reference guide. ChatGPT has been used to enhance the code to ensure high quality content in some cases.

## Virtual Environment

Dependencies are listed here: [requirements.txt](requirements.txt)

On Windows, they can be installed with below commands:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: Package versions were not specified in requirements.txt to allow pip to resolve conflicts. This may break some of the code's reproducibility in the future. When GPU based execution is preferred for PyTorch based code implementations, PyTorch installation in requirements.txt should be removed and version of PyTorch with Cuda should be installed manually.

## [Paper to Code](paper_to_code)

Implementation of academic papers with relevant examples of applications.

* [Cadence Time Series Partitioning](paper_to_code/cadence_time_series_partitioning/):  Chowdhury, T., Aldeer, M., Laghate, S., et al. (2021). Time Series Segmentation Using Autoencoders. arXiv preprint arXiv:2112.03360. [DOI: 10.48550/arXiv.2112.03360](https://doi.org/10.48550/arXiv.2112.03360)


## [Tree Based Models](tree_based_models)

Collection of regression/classification problems using tree based models such as:

* Decision Tree
* Random Forest
* ADA Boost
* Gradient Boost
* Extreme Gradient Boost (XGBoost)

## [Physics Informed Neural Networks](physics_informed_neural_networks)

Solutions to Partial Differential Equations in Physics such as Heat Transfer and Fluid Dynamics using both numerical methods and Physics Informed Neural Networks. 

These are modified versions of examples from the Udemy Class (https://www.udemy.com/course/physics-informed-neural-network-pinns/) by Dr. Mohammad Samara.

* 1D Heat Equation Numerical Methods
* 2D Burgers Equation Numerical Methods
* 1D Burgers Equation PINN & PyTorch
* 1D Heat Equation PINN & DeepXDE Library

Note: Code has been enhanced & documented with help of ChatGPT.

## [Functional Data Analysis](functional_data_analysis)

Contains several methods in functional data analysis such as splines, kernel smoothers and functional principal component analysis (FPCA).

* B Spline Regression
* Smoothing Spline Regression
* Natural Cubic Spline Regression
* Kernel Smoother Regression
* Kernel Smoother Local Linear Regression
* Kernel Smoother Local Polynomial Regression
* ECG Heartbeat Categorization problem, implementing multiple classifications algorithms with and without B-spline transformation to deal with high dimensional data.

Examples are conversion of Matlab/R based lecture notes from Georgia Tech's ISYE 8803 High Dimensional Data Analytics class. In some parts, ChatGPT has been used for conversion to Python as well as documentation generation.

## [Change Point Detection](change_point_detection)

Evaluation of different change point detection algorithms.

* CUSUM (Cumulative Sum Control Chart)


