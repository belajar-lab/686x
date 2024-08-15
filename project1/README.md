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

### Average Perceptron Algorithm
The average perceptron will add a modification to the original perceptron algorithm: since the basic algorithm continues updating as the algorithm runs, nudging parameters in possibly conflicting directions, it is better to take an average of those parameters as the final answer. Every update of the algorithm is the same as before. The returned parameters $\theta$, however, are an average of the $\theta$s across the $nT$ steps:

$$
\theta _{final} = \frac{1}{nT}(\theta ^{(1)} + \theta ^{(2)} + ... + \theta ^{(nT)})
$$

You will now implement the average perceptron algorithm. This function should be constructed similarly to the Full Perceptron Algorithm above, except that it should return the average values of $\theta$ and $\theta_0$.

>[!Tip]
>Tracking a moving average through loops is difficult, but tracking a sum through loops is simple.

>[!Note]
>Please call `get_order(feature_matrix.shape[0])`, and use the ordering to iterate the feature matrix in each iteration. In practice, people typically just randomly shuffle indices to do stochastic optimization.


## Pegasos Algorithm

Now you will implement the Pegasos algorithm. For more information, refer to the original paper at [original paper](https://www.notion.so/Automatic-Review-Analyzer-fa12e75898404964aeca1ad1f41db923?pvs=21).

The following pseudo-code describes the Pegasos update rule.

$$
\begin{align*}
&\textmd{Pegasos update rule}\ \left(x^{(i)}, y^{(i)}, \lambda , \eta , \theta \right):\\

&\kern1.5em \textbf{if}\ y^{(i)}(\theta \cdot x^{(i)}) \leq 1 \ \textbf{then}\\

&\kern3em \textbf{update}\ \theta = (1 - \eta \lambda ) \theta + \eta y^{(i)}x^{(i)}\\

&\kern1.5em\textbf{else}:\\

&\kern3em \textbf{update}\ \theta = (1 - \eta \lambda ) \theta
\end{align*}
$$

The $\eta$ parameter is a decaying factor that will decrease over time. The $\lambda$ parameter is a regularizing parameter.

In this problem, you will need to adapt this update rule to add a bias term ($\theta_0$) to the hypothesis, but take care not to penalize the magnitude of $\theta_0$.

### Pegasos Single Step Update
Next you will implement the single step update for the Pegasos algorithm. This function is very similar to the function that you implemented in **Perceptron Single Step Update**, except that it should utilize the Pegasos parameter update rules instead of those for perceptron. The function will also be passed a $\lambda$ and $\eta$ value to use for updates.

### Full Pegasos Algorithm
Finally you will implement the full Pegasos algorithm. You will be given the same feature matrix and labels array as you were given in **Full Perceptron Algorithm**. You will also be given $T$, the maximum number of times that you should iterate through the feature matrix before terminating the algorithm. Initialize $\theta$ and $\theta _0$ to zero. For each update, set $\displaystyle \eta = \frac{1}{\sqrt{t}}$ where $t$ is a counter for the number of updates performed so far (between $1$ and $nT$ inclusive). This function should return a tuple in which the first element is the final value of $\theta$ and the second element is the value of $\theta _0$.

>[!Note]
>Please call `get_order(feature_matrix.shape[0])`, and use the ordering to iterate the feature matrix in each iteration. In practice, people typically just randomly shuffle indices to do stochastic optimization.


## Algorithm Discussion

Once you have completed the implementation of the 3 learning algorithms, you should qualitatively verify your implementations. In *main.py* we have included a block of code that you should uncomment. This code loads a 2D dataset from *toy_data.txt*, and trains your models using $T = 10, \lambda = 0.2$. *main.py* will compute $\theta$ and $\theta _0$ for each of the learning algorithms that you have written. Then, it will call `plot_toy_data` to plot the resulting model and boundary.

### Plots
In order to verify your plots, please enter the values of $\theta$ and $\theta_0$ for all three algorithms (perceptron, average perceptron, Pegasos).

### Convergence
Since you have implemented three different learning algorithm for linear classifier, it is interesting to investigate which algorithm would actually converge. Please run it with a larger number of iterations $T$ to see whether the algorithm would visually converge. You may also check whether the parameter in your theta converge in the first decimal place. Achieving convergence in longer decimal requires longer iterations, but the conclusion should be the same.

Which of the algorithm will converge on this dataset?


## Automative review analyzer

Now that you have verified the correctness of your implementations, you are ready to tackle the main task of this project: building a classifier that labels reviews as positive or negative using text-based features and the linear classifiers that you implemented in the previous section!

## The Data
The data consists of several reviews, each of which has been labeled with $-1$ or $+1$, corresponding to a negative or positive review, respectively. The original data has been split into four files:

- `reviews_train.tsv` (4000 examples)
- `reviews_validation.tsv` (500 examples)
- `reviews_test.tsv` (500 examples)

To get a feel for how the data looks, we suggest first opening the files with a text editor, spreadsheet program, or other scientific software package (like [pandas](https://pandas.pydata.org/)).

### Translating reviews to feature vectors
We will convert review texts into feature vectors using a **bag of words** approach. We start by compiling all the words that appear in a training set of reviews into a **dictionary**, thereby producing a list of $d$ unique words.

We can then transform each of the reviews into a feature vector of length $d$ by setting the $i^{th}$ coordinate of the feature vector to $1$ if the $i^{th}$ word in the dictionary appears in the review, or $0$ otherwise. For instance, consider two simple documents “Mary loves apples” and “Red apples”. In this case, the dictionary is the set $\{ \text {Mary}; \text {loves}; \text {apples}; \text {red}\}$, and the documents are represented as $(1; 1; 1; 0)$ and $(0; 0; 1; 1)$.

A bag of words model can be easily expanded to include phrases of length $m$. A **unigram** model is the case for which $m=1$. In the example, the unigram dictionary would be $(\text {Mary}; \text {loves}; \text {apples}; \text {red})$. In the **bigram** case, $m=2$, the dictionary is $(\text {Mary loves}; \text {loves apples}; \text {Red apples})$, and representations for each sample are $(1; 1; 0), (0; 0; 1)$. In this section, you will only use the unigram word features. These functions are already implemented for you in the `bag_of_words`  function.

In `utils.py`, we have supplied you with the `load data` function, which can be used to read the `.tsv` files and returns the labels and texts. We have also supplied you with the `bag_of_words` function in `project1.py`, which takes the raw data and returns dictionary of uni-gram words. The resulting dictionary is an input to `extract_bow_feature_vectors` which you will edit to compute a feature matrix of ones and zeros that can be used as the input for the classification algorithms. Using the feature matrix and your implementation of learning algorithms from before, you will be able to compute $\theta$ and $\theta_0$.


## Classification and Accuracy

Now we need a way to actually use our model to classify the data points. In this section, you will implement a way to classify the data points using your model parameters, and then measure the accuracy of your model.

### Classification
Implement a classification function that uses $\theta$ and $\theta_0$ to classify a set of data points. You are given the feature matrix, $\theta$, and $\theta_0$ as defined in previous sections. This function should return a numpy array of -1s and 1s. If a prediction is **greater than** zero, it should be considered a positive classification.

>[!Tip]
>As in previous exercises, when $x$ is a float, “$x=0$” should be checked with $|x| < \epsilon$.

### Accuracy
We have supplied you with an `accuracy` function:

```python
def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
```

The `accuracy` function takes a numpy array of predicted labels and a numpy array of actual labels and returns the prediction accuracy. You should use this function along with the functions that you have implemented thus far in order to implement `classifier_accuracy`.

The `classifier_accuracy` function should take 6 arguments:

- a classifier function that, itself, takes arguments `(feature_matrix, labels, **kwargs)`
- the training feature matrix
- the validation feature matrix
- the training labels
- the validation labels
- a `**kwargs` argument to be passed to the classifier function

This function should train the given classifier using the training data and then compute compute the classification accuracy on both the train and validation data. The return values should be a tuple where the first value is the training accuracy and the second value is the validation accuracy.

### Baseline Accuracy
Now,

- Edit the function `extract_bow_feature_vectors` under `if binarize:` in *project1.py* so that the feature matrix encodes the presence of a word as  if present, and  if not (i.e. one-hot encoding),
- uncomment the relevant lines in *main.py*,
- report the training and validation accuracies of each algorithm with T = 10 and  = 0.01 (the  value only (applies to Pegasos).

>[!Hint]
>If you get the following warnings when runnning *test.py*, do not worry:

```
WARN Bag of words : does not remove stopwords: ['he', 'is', 'on', 'the', 'there', 'to'] 
lWARN Extract bow feature vectors : uses binary indicators as features
```

Please enter the **validation accuracy** of your Perceptron, Average Perceptron, and Pegasos algorithm.