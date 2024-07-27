# Analytics Journal

Collection of different topics in data science & computational methods and their implementation examples.

These serves as my notes towards general analytics applications, to be used as quick reference guide. ChatGPT has been used to enhance the code to ensure high quality content in some cases.

## Virtual Environment

Dependencies are listed under requirements.txt of each directory.

On Windows, they can be installed with below commands using a virtual environment:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Docker Containers

Below are useful multi-purpuse Docker files images.

### [PySpark, HDFS and Jupyter Lab Server](docker_images/pyspark_jupyter_server/docker-compose.yaml)

Below number of workers can be adjusted:
```
cd docker_images\pyspark_jupyter_server
docker-compose up --scale spark-worker=2
```

To List Containers Including Stopped Ones:
```
docker ps -a
```

To Start Container:
```
docker start {container_name}
```

Load Jupyter Lab:
```
http://localhost:8888/lab
```

Load Spark View:
```
http://localhost:8080/
```

[Demo Notebook](docker_images/pyspark_jupyter_server/demo.ipynb)

## Topics

### [Paper to Code](paper_to_code)

Implementation of academic papers with relevant examples of applications.

* [Cadence Time Series Partitioning](paper_to_code/cadence_time_series_partitioning/):  Chowdhury, T., Aldeer, M., Laghate, S., et al. (2021). Time Series Segmentation Using Autoencoders. arXiv preprint arXiv:2112.03360. [DOI: 10.48550/arXiv.2112.03360](https://doi.org/10.48550/arXiv.2112.03360)


### [Tree Based Models](tree_based_models)

Collection of regression/classification problems using tree based models such as:

* Decision Tree
* Random Forest
* ADA Boost
* Gradient Boost
* Extreme Gradient Boost (XGBoost)

### [Physics Informed Neural Networks](physics_informed_neural_networks)

Solutions to Partial Differential Equations in Physics such as Heat Transfer and Fluid Dynamics using both numerical methods and Physics Informed Neural Networks. 

These are modified versions of examples from the Udemy Class (https://www.udemy.com/course/physics-informed-neural-network-pinns/) by Dr. Mohammad Samara.

* 1D Heat Equation Numerical Methods
* 2D Burgers Equation Numerical Methods
* 1D Burgers Equation PINN & PyTorch
* 1D Heat Equation PINN & DeepXDE Library

Note: Code has been enhanced & documented with help of ChatGPT.

### [Functional Data Analysis](functional_data_analysis)

Contains several methods in functional data analysis such as splines, kernel smoothers and functional principal component analysis (FPCA).

* B Spline Regression
* Smoothing Spline Regression
* Natural Cubic Spline Regression
* Kernel Smoother Regression
* Kernel Smoother Local Linear Regression
* Kernel Smoother Local Polynomial Regression
* ECG Heartbeat Categorization problem, implementing multiple classifications algorithms with and without B-spline transformation to deal with high dimensional data.

Examples are conversion of Matlab/R based lecture notes from Georgia Tech's ISYE 8803 High Dimensional Data Analytics class. In some parts, ChatGPT has been used for conversion to Python as well as documentation generation.

### [Change Point Detection](change_point_detection)

Evaluation of different change point detection algorithms.

* CUSUM (Cumulative Sum Control Chart)

### [Optimization](optimization)

Containts examples of different type of optimization problems and its problem solutions via different Python packages.

* Linear Program
* Mixed Integer Program
* Mixed Integer Quadratic Program
* Non Linear Program
* Mixed Integer Non Linear Program

## Install Optimization Solvers

Optimization section requires installation of solvers. List of popular free solvers are below:

### GLPK (GNU Linear Programming Kit)

* Download GLPK from SourceForge: https://sourceforge.net/projects/winglpk/
* Unzip glpk-4.65 folder and place it in ```C:\glpk-4.65```
* Add ```C:\glpk-4.65\w64``` to PATH if using x64
* Test that its working on CMD and get the executable path: ```where glpsol```
* Usage with Pyomo: ```solver = SolverFactory("glpk", executable= r"C:\glpk-4.65\w64\glpsol.exe")```

### IPOPT (Interior Point OPTimizer)

* Download IPOPT from SourceForge: https://www.coin-or.org/download/binary/Ipopt/Ipopt-3.11.1-win64-intel13.1.zip
* Unzip Ipopt-3.11.1-win64-intel13.1 folder and place it in ```C:\Ipopt-3.11.1-win64-intel13.1```
* Add ```C:\Ipopt-3.11.1-win64-intel13.1\bin``` to PATH
* Test that its working on CMD and get the executable path: ```where ipopt```
* Usage with Pyomo: ```solver = SolverFactory("ipopt", executable= r"C:\Ipopt-3.11.1-win64-intel13.1\bin\ipopt.exe")```

### CPLEX (C Programming Language for EXecution.)

* Download and install IBM CPLEX Optimization Studio: https://www.ibm.com/account/reg/signup?formid=urx-20028
* Add ```C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2211\cplex\bin\x64_win64``` to PATH
* Test that its working on CMD and get the executable path: ```where cplex```
* Usage with Pyomo: ```solver = SolverFactory("cplex", executable= r"C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2211\cplex\bin\x64_win64\cplex.exe")```

### CBC (COIN-OR Branch and Cut)

* Download CBC Solver: https://github.com/coin-or/Cbc/releases/download/releases%2F2.10.11/Cbc-releases.2.10.11-w64-msvc17-md.zip
* Unzip glpk-4.65 folder and place it in ```:\Cbc-releases.2.10.11-w64-msvc17-md```
* Add ```C:\Cbc-releases.2.10.11-w64-msvc17-md\bin``` to PATH
* Test that its working on CMD and get the executable path: ```where cbc```
* Usage with Pyomo: ```solver = SolverFactory("cbc", executable= r"C:\Cbc-releases.2.10.11-w64-msvc17-md\bin\cbc.exe")```

### SCIP

* Download and install SCIP Optimization Suite: https://scipopt.org/download.php?fname=SCIPOptSuite-9.1.0-win64-VS15.exe
* Add ```C:\Program Files\SCIPOptSuite 9.1.0\bin\``` to PATH
* Test that its working on CMD and get the executable path: ```where scip```
* Usage with Pyomo: ```solver = SolverFactory("scip", executable= r"C:\Program Files\SCIPOptSuite 9.1.0\bin\scip.exe")```