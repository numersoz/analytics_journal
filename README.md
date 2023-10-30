# Data Science Journal

Collection of different topics in Data Science and their implementation examples.

## Virtual Environment

Dependencies are listed here: [requirements.txt](requirements.txt)

On Windows, they can be installed with below commands:

```
python3 -m venv .venv
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

Note: Code has been beautified & documented with help of ChatGPT.
