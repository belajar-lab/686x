## Introduction

The goal of this project 0 is to help you set up your python environment and to give you an introduction to the mechanics of the grading system.

As a reminder, this course assumes a basic understanding of Python at the level of [Introduction to Computer Science and Programming Using Python](https://www.edx.org/course/introduction-to-computer-science-and-programming-using-python-2) (also on [MIT Open CourseWare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/)). This means you should be proficient in the following: functions, tuples and lists, mutability, recursion, dictionaries, and object-oriented programming.

Additionally, we expect you to be able to install the required packages using [`pip`](https://pip.pypa.io/en/stable/installing/) and be comfortable reading the documentation of these packages to find out more about the functions you are not familiar with.

![1: https://xkcd.com/1987/](https://courses.edx.org/asset-v1:MITx+6.86x+2T2024+type@asset+block/images_python_environment.png)


## Setting up packages

### Required python packages

Throughout this course, we will be using Python 3.8 along with the following packages. Code written in new versions of python will be accepted, as long as functions/features that are available only in Python 3.9 or beyond are not used.

- [NumPy](https://www.numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [SciPy](https://www.scipy.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [PyTorch](https://pytorch.org/)

### Installation using pip

If you already have a working installation of Python 3, you should be able to install all of the above packages using pip.

```sh
pip3 install numpy
pip3 install matplotlib
pip3 install scipy
pip3 install tqdm
pip3 install scikit-learn
```

For PyTorch, follow the instructions on https://pytorch.org/ to install from pip repository corresponding to your system. You will not need CUDA for this course.

In the above commands, you can replace `pip3` with `python3 -m pip` to make sure you are installing the packages for the version of python your system is currently using.

### Installation using conda

However, the recommended way of configuring your system is by using a conda environment.

We recommend that you install the latest version of https://docs.anaconda.com/miniconda/.

You can then create a conda environment for this course using

```sh
conda create -n 6.86x python=3.8
```

> [!NOTE]
>As mentioned above, you may use other versions of python, as long as functions available only in 3.9+ are not used.

To activate this environment, use
```sh
conda activate 6.86x
```

Finally, install all of the required packages:
```sh
conda install pytorch -c pytorch
conda install numpy
conda install matplotlib
conda install scipy
conda install tqdm
conda install scikit-learn
```

### Creating Environments Using Anaconda
[Creating Environments Using Anaconda](https://edx-video.net/MITx6.86x-V015300_DTH.mp4)


## Testing your installation

The `project0` folder contains two python files.

- *main.py* contains the various functions you will to complete in the next sections of the project
- *test.py* is a script which runs tests
- *debug.py* contains the code for the final problem of this project

> [!TIP]
>Throughout the whole project, you can assume the NumPy python library is already imported as `np`.

You are welcome to implement functions locally to fully check correctness and individual function implementations. 

**How to Test:** In your terminal, navigate to the directory where your project files reside. 
Execute the command `python test.py` to run all the available tests.

For this project, the `test.py` file will test that all required packages are correctly installed.

> [!TIP]
>We recommend using a proper IDE for this course such as Visual Studio Code, Pycharm, etc.


## Introduction to Numpy

Here, we introduce Numpy, a mathematics framework for Python.
- [Fundamentals of Numpy](https://edx-video.net/MITx6.86x-V003100_DTH.mp4)

### Randomization

Write a function called `randomization` that takes as input a positive integer `n`, and returns `A`, a random $n \times 1$ Numpy array.

### Operations

Here, we learn how to find dimensions and apply matrix multiplication using Numpy arrays.
- [Matrix Properties and Operations](https://edx-video.net/MITx6.86x-V004400_DTH.mp4)

Write a function called `operations` that takes as input two positive integers `h` and `w`, makes two random matrices `A` and `B`, of size $h \times w$, and returns `A`,`B`, and `s`, the sum of `A` and `B`.

### Norm

Here, we learn how to find the maximum and minimum values in a Numpy array and obtain the norm of a vector.
- [Max, Min, and Norm](https://edx-video.net/MITx6.86x-V004500_DTH.mp4)

Write a function called `norm` that takes as input two Numpy column arrays `A` and `B`, adds them, and returns `s`, the L2 norm of their sum.


## Exercise

As introduced in the previous section, a neural network is a powerful tool often utilized in machine learning. Because neural networks are, fundamentally, very mathematical, we'll use them to motivate Numpy!

We review the simplest neural network here:

<img src="https://courses.edx.org/asset-v1:MITx+6.86x+2T2024+type@asset+block/images_basic_nn.png" width="256">

The output of the neural network, $z_1$, is dependent on the inputs $x_1$  and $x_2$ . The importance of each of the inputs is given by values called *weights*. There is one weight from each input to each output. We show this here:

<img src="https://courses.edx.org/asset-v1:MITx+6.86x+2T2024+type@asset+block/images_basic_nn_weights.png" width="256">

The inputs are given by $x$, and the outputs are given by $z_1$. Here, $w_{11}$ is the weight of input 1 on output 1 (our only output in this case), and $w_{21}$ is the weight of input 2 on output 1. In general, $w_{ij}$ represents the weight of input $i$ on output $j$.

The output, $z_1$, is given by $z_1 = f(w_{11}x_1 + w_{21}x_2)$:

<img src="https://courses.edx.org/asset-v1:MITx+6.86x+2T2024+type@asset+block/images_basic_nn_product1.png" width="512">

where $f$ is a specified nonlinear function, and it is usually the hyperbolic tangent function, $\tanh$.

If we express our inputs and weights as matrices, as shown here,

$$
\vec{x} = \begin{bmatrix}x_1 \\ x_2\end{bmatrix} \quad w= \begin{bmatrix}w_{11} \\ w_{21}\end{bmatrix} 
$$

then we can develop an elegant mathematical expression: $z_1 = \tanh (w^{T}\vec{x})$.

### Neural Network

Here, we will write a function `neural_network`, which will apply a neural network operation with 2 inputs and 1 output and a given weight matrix.

Your function should take two arguments: `inputs` and `weights`, two NumPy arrays of shape $(1,2)$ and should return a NumPy array of shape $(1,1)$, the output of the neural network. Do not forget the $\tanh$ activation.


## Vectorize function

In this exercise, you will learn how to vectorize a function that can only deal with scalar inputs without using a `for` loop.

### Scalar function

Let's start with writing a scalar function `scalar_function`, which will apply the following operation with input `x` and `y`.

$$
f(x,y)=\begin{cases}  x\cdot y, \text { if } x\le y\\ x/y, \text { else.} \end{cases}
$$

Note that `x` and `y` are scalars.

### Vector function

`scalar_function` can only handle scalar input, we could use the function `np.vectorize()` turn it into a vectorized function. Note that the input argument of `np.vectorize()` should be a scalar function, and the output of `np.vectorize()` is a new function that can handle vector input.

Please write a vector function `vector function`, which will apply the operation $f(x,y)$ defined above element-wisely with input vectors with same dimension `x` and `y`.


## Introduction to ML packages

In the resources, we have provided you with two notebooks.

### Introduction to ML packages (part 1) 
- [Github](https://github.com/Varal7/ml-tutorial/blob/master/Part1.ipynb)
- [Notebook viewer](https://nbviewer.jupyter.org/github/Varal7/ml-tutorial/blob/master/Part1.ipynb)

### Introduction to ML packages (part 2)
- [Github](https://github.com/Varal7/ml-tutorial/blob/master/Part2.ipynb)
- [Notebook viewer](https://nbviewer.jupyter.org/github/Varal7/ml-tutorial/blob/master/Part2.ipynb)

They cover some of the most useful ML packages and constitute a good reference point to refer to as you progress through the course.

We do not expect you to complete all sections in these notebooks immediately. For now, go through the first three sections in the first notebook on **Jupyter**, **Numpy**, and **Matplotlib**. Then after Unit 1 *Linear Classifiers*, come back to the section on **Scikit learn**, and while you work on Unit 3 *Neural Nets*, refer to the second notebook, which gives an introduction to **Pytorch**.

We will not be using **Pandas** in this course, but it is a useful tool. Feel free to look at the section on Pandas at any time.


## Debugging

In machine learning, there a lot of reasons why a model would fail to perform well. The model could be poorly designed, the data could be too noisy, it could just be a poor initialization, or a poor choice of hyperparameters, etc. Therefore, it is vital to at least exclude engineering bugs using a debugger.

Fortunately, Python comes with a fully functional interactive debugger called [pdb](https://docs.python.org/3/library/pdb.html).

Before you tackle the last problem of this project, we recommend you take some time to learn how to use, for example with [this tutorial](https://realpython.com/python-debugging-pdb/).

### Debugging exercise

In this problem, you are given a buggy piece of code and are asked to debug it.

The goal of this exercise is for you to set up a working debugging system for yourselves. (See the next page for an example.) Feel free to use other debuggers as you wish. But note that any extra print statement in submitted code will be graded as incorrect in the online code graders.

The function `get_sum_metrics` takes two arguments: a `prediction` and a list of `metrics` to apply to the prediction (say, for instance, the accuracy or the precision). Note that each metric is a function, not a number. The function should compute each of the metrics for the prediction and sum them. It should also add to this sum three default metrics, in this case, adding 0, 1 or 2 to the prediction.

> [!IMPORTANT]
>You should fix this function first, and run `python debug.py` in your `project0` directory to make sure it behaves as expected on the simple test cases provided


## IDE and Debugger Demo
- [Introduction](https://edx-video.net/MITx6.86x-V015000_DTH.mp4)
- [Pycharm IDE](https://edx-video.net/MITx6.86x-V015200_DTH.mp4)
- [Debugging](https://edx-video.net/MITx6.86x-V015100_DTH.mp4)