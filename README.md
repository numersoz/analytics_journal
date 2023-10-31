# Analytics Journal

Collection of different topics in data science & computational methods and their implementation examples.

These serves as my notes towards general analytics applications, to be used as quick reference guide. Beware, ChatGPT has been used to enhance the code to ensure high quality content.

## Virtual Environment

Dependencies are listed here: [requirements.txt](requirements.txt)

On Windows, they can be installed with below commands:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Note: Package versions were not specified in requirements.txt to allow pip to resolve conflicts. This may break some of the code's reproducibility in the future.

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
