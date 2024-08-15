## Introduction

The goal of this project is to design a classifier to use for sentiment analysis of product reviews. Our training set consists of reviews written by Amazon customers for various food products. The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively.

Below are two example entries from our dataset. Each entry consists of the review and its label. The two reviews were written by different customers describing their experience with a sugar-free candy.

| Review | label |
| --- | --- |
| Nasty No flavor. The candy is just red, No flavor. Just plan and chewy. I would never buy them again | $-1$ |
| YUMMY! You would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free! i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office. These are just EXCELLENT! | $1$ |

In order to automatically analyze reviews, you will need to complete the following tasks:

1. Implement and compare three types of linear classifiers: the **perceptron** algorithm, the **average perceptron** algorithm, and the **Pegasos** algorithm.
2. Use your classifiers on the food review dataset, using some simple text features.
3. Experiment with additional features and explore their impact on classifier performance.

### Setup Details

For this project and throughout the course we will be using Python 3 with some additional libraries. We strongly recommend that you take note of how the NumPy numerical library is used in the code provided, and read through the on-line NumPy tutorial. **NumPy arrays are much more efficient than Python's native arrays when doing numerical computation. In addition, using NumPy will substantially reduce the lines of code you will need to write.**

1. *Note on software: For this project, you will need the **NumPy** numerical toolbox, and the **matplotlib** plotting toolbox.*
2. The `sentiment_analysis` folder contains the various data files in *.tsv* format, along with the following python files:
   - *project1.py* contains various useful functions and function templates that you will use to implement your learning algorithms.
   - *main.py* is a script skeleton where these functions are called and you can run your experiments.
   - *utils.py* contains utility functions that the staff has implemented for you.
   - *test.py* is a script which runs tests on a few of the methods you will implement. These tests are provided to help you debug your implementation. Feel free to add more test cases locally to further validate the correctness of your code.

**How to Test:** In your terminal, navigate to the directory where your project files reside. Execute the command `python test.py` to run all the available tests.

**How to Run your Project 1 Functions:** In your terminal, enter `python main.py`. You will need to uncomment/comment the relevant code as you progress through the project.

>[!Tip]
>You may also first go through the recitation at the end of this unit before or concurrently with this project.


## Hinge Loss

In this project you will be implementing linear classifiers beginning with the Perceptron algorithm. You will begin by writing your loss function, a hinge-loss function. For this function you are given the parameters of your model $\theta$ and $\theta _0$. Additionally, you are given a feature matrix in which the rows are feature vectors and the columns are individual features, and a vector of labels representing the actual sentiment of the corresponding feature vector.

### Hinge Loss on One Data Sample
First, implement the basic hinge loss calculation on a single data-point. Instead of the entire feature matrix, you are given one row, representing the feature vector of a single data sample, and its label of +1 or -1 representing the ground truth sentiment of the data sample.

### The Complete Hinge Loss
Now it's time to implement the complete hinge loss for a full set of data. Your input will be a full feature matrix this time, and you will have a vector of corresponding labels. The $k^{th}$ row of the feature matrix corresponds to the $k^{th}$ element of the labels vector. This function should return the appropriate loss of the classifier on the given dataset.


## Perceptron Algorithm

Now you will implement the Perceptron algorithm

### Perceptron Single Step Update
Now you will implement the single step update for the perceptron algorithm (implemented with $0-1$ loss). You will be given the feature vector as an array of numbers, the current $\theta$ and $\theta_0$ parameters, and the correct label of the feature vector. The function should return a tuple in which the first element is the correctly updated value of $\theta$ and the second element is the correctly updated value of $\theta_0$.

>[!Tip]
>Because of numerical instabilities, it is preferable to identify $0$ with a small range $[-\varepsilon , \varepsilon ]$. That is, when $x$ is a float, "$x=0$" should be checked with $|x| < \varepsilon$.

### Full Perceptron Algorithm
In this step you will implement the full perceptron algorithm. You will be given the same feature matrix and labels array as you were given in **The Complete Hinge Loss**. You will also be given $T$, the maximum number of times that you should iterate through the feature matrix before terminating the algorithm. Initialize $\theta$ and $\theta_0$ to zero. This function should return a tuple in which the first element is the final value of $\theta$ and the second element is the value of $\theta_0$.

>[!Tip]
>Call the function `perceptron_single_step_update` directly without coding it again.

>[!Hint] 
>Make sure you initialize `theta` to a 1D array of shape `(n,)` and **not** a 2D array of shape `(1, n)`.

>[!Note]
>Please call `get_order(feature_matrix.shape[0])`, and use the ordering to iterate the feature matrix in each iteration. In practice, people typically just randomly shuffle indices to do stochastic optimization.