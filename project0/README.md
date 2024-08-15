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