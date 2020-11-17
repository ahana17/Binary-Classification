# Python-Project

The training dataset:https://drive.google.com/file/d/1UJrpHQiJDMpsh-_utV14WoR-YR6zjoD2/view?usp=sharing
The test dataset:https://drive.google.com/file/d/16aLIOB6Lwv_wHJwGnorqDkDeofX2hj4V/view?usp=sharing

1. Logistic Regression
Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.
Logistic regression uses a sigmoid function to predict the output. The sigmoid function returns a value from 0 to 1. Generally, we take a threshold such as 0.5. If the sigmoid function returns a value greater than or equal to 0.5, we take it as 1, and if the sigmoid function returns a value less than 0.5, we take it as 0.
                        

Python Output:
 


Model score = 0.9
Accuracy = 0.832
Training error = 1-model score = 0.1
Misclassification error = 1-accuracy = 0.168

The ROC curve was plotted.

Other evaluation parameters for this model include:

F1 Score = 0.83
Recall score = 0.813
Area under ROC curve(AUC) = 0.87
The model can be viewed at:
https://drive.google.com/file/d/1FB9dZNt6Xyiqgmur1yt_MIQB_AbpvyDA/view?usp=sharing




2. K-Nearest Neighbors (KNN) algorithm
K in KNN is the number of nearest neighbors considered for assigning a label to the current point. K is an extremely important parameter and choosing the value of K is the most critical problem when working with the KNN algorithm. The process of choosing the right value of K is referred to as parameter tuning and is of great significance in achieving better accuracy. If the value of K is too small then there is a probability of overfitting the model and if it is too large then the algorithm becomes computationally expensive. Most data scientists usually choose an odd number value for K when the number of classes is 2.  Another formula that works well for choosing K is, k- sqrt(n) where n is the total number of data points.
Selecting the value of K depends on individual cases and sometimes the best method of choosing K is to run through different values of K and verify the outcomes. Using cross-validation, the KNN algorithm can be tested for different values of K and the value of K that results in good accuracy can be considered as an optimal value for K
                                  
Python Output:
 
First data is fitted without scaling and accuracy(=1-misclassification error) and the training score is compared for different values of K.

Error rate vs K-value graph is plotted and the best accuracy is obtained at k=85

k=1->          acc-0.61,       training error=0.0 ; 
k=40->        acc=0.75;
k=85->        acc=.773,      training error=0.23




At k=85


The input data is scaled and the same procedure is repeated.
Best accuracy is obtained at k=78 from error vs k-value graph as well as from calculations.
k=1->         acc-0.61,           training error=0.0; 
k=40->       acc=0.76;        
k=78->       acc=.785,          training error=0.235
At k=78,


The ROC curve was plotted.
Other evaluation parameters for this model include:

F1 Score = 0.785;
Recall score = 0.772;
AUC = 0.84
Misclassification error = 0.215;
Training error = 0.235

The model can be viewed at:
https://drive.google.com/file/d/1em-YOmVAWty__7qiRiDEKNlN8J7v8RcA/view?usp=sharing

3. Naive Bayes
Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred. Bayes’ theorem is stated mathematically as the following equation:

Now, with regards to our dataset, we can apply Bayes’ theorem in the following way:

where y is class variable and X is a dependent feature vector (of size n) where:

 
the features are assumed to be independent. Hence, we reach the result:

which can be expressed as:

Now, as the denominator remains constant for a given input, we can remove that term:

Now, we need to create a classifier model. For this, we find the probability of a given set of inputs for all possible values of the class variable y and pick up the output with maximum probability. This can be expressed mathematically as:

Gaussian Naive Bayes 
In Gaussian Naive Bayes, continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution. A Gaussian distribution is also called Normal distribution. When plotted, it gives a bell-shaped curve which is symmetric about the mean of the feature values as shown below:

The likelihood of the features is assumed to be Gaussian, hence, conditional probability is given by:


 
Python Output:
 
                    
Accuracy = 0.794,
Training error = 0.125,
Misclassification error = 0.206

The ROC curve was plotted.
Other evaluation parameters for this model include:    

F1 Score = 0.791; 
Recall score = 0.768;
AUC = 0.85

The model can be viewed at:
https://drive.google.com/file/d/1Df1wYiuV6--sPt8v1iNyVnEh5DQ7qPAR/view?usp=sharing


4.  Support vector machine (SVM)
     We can use a support vector machine (SVM) when our data has exactly two classes. An SVM classifies data by finding the best hyperplane that separates all data points of one class from those of the other class. The best hyperplane for an SVM means the one with the largest margin between the two classes. Margin means the maximal width of the slab parallel to the hyperplane that has no interior data points.
The support vectors are the data points that are closest to the separating hyperplane; these points are on the boundary of the slab. The following figure illustrates these definitions, with + indicating data points of type 1, and – indicating data points of type 0.



Python Output:
 
1. Simple SVM


Accuracy = 0.817,
Training error = 0.04,
Misclassification error = 0.183

The ROC curve was plotted.
Other evaluation parameters for this model include:

F1 Score = 0.817;
Recall score = 0.802;
AUC = 0.86

2. Kernel SVM

(a) Polynomial Kernel


Accuracy = 0.742,
Training error = 0.0,
Misclassification error = 0.258

The ROC curve was plotted.
Other evaluation parameters for this model include:

F1 Score = 0.725;
Recall score = 0.668;
AUC = 0.82

(b) Gaussian Kernel


Accuracy = 0.817,
Training error=0.04,
Misclassification error = 0.183
The ROC curve was plotted.
Other evaluation parameters for this model include:

F1 Score = 0.817;
Recall score = 0.802;
AUC = 0.86

(c) Sigmoid Kernel


Accuracy = 0.475,
Training error = 0.535,
Misclassification error = 0.525
The ROC curve was plotted.
Other evaluation parameters for this model include:

F1 Score = 0.093;
Recall score = 0.053;
AUC = 0.59

The model can be viewed at:
https://drive.google.com/file/d/1ZVZ2IODB9krXw8s9LnoA5bhm3_dTHXJU/view?usp=sharing



5. Convolutional Neural Network (CNN)
A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with multiplication or other dot product. The activation function is commonly a RELU layer and is subsequently followed by additional convolutions such as pooling layers, fully connected layers, and normalization layers referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.

Python Output:
 
classifier = Sequential()

Accuracy = 0.811,
Training error = 0.055,
Misclassification error = 0.189

The ROC curve was plotted.
Other evaluation parameters for this model include:

F1 Score = 0.808;
Recall score = 0.782;
AUC = 0.86


The model can be viewed at:
https://drive.google.com/file/d/1GYCzk8AmPr8dpg8CIJNyBsiUzPDL-SLp/view?usp=sharing


6. Multilayer perceptron neural network (MLP)
An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

Python Output:

1. Without Scaling data
The solver for weight optimization- ‘lbfgs’ is an optimizer in the family of quasi-Newton methods;
hidden_layer_sizes = (5, 2)


Accuracy = 0.796,
Training error = 0.025,
Misclassification error = 0.204

The ROC curve was plotted.
Other evaluation parameters for this model include:

F1 Score = 0.784;
Recall score = 0.729;
AUC = 0.76

2. Scaling Data and Controlling with warm start

hidden_layer_sizes = (13,10,15) and maximum iteration = 1000

Accuracy = 0.757,
Training error = 0.0,
Misclassification error = 0.243

The ROC curve was plotted.
Other evaluation parameters for this model include:

F1 Score = 0.757;
Recall score = 0.745;
AUC = 0.81

The model can be viewed at:
https://drive.google.com/file/d/1Df1wYiuV6--sPt8v1iNyVnEh5DQ7qPAR/view?usp=sharing
 
7. XGBoost
XGBoost is one of the most popular and efficient implementations of the Gradient Boosted Trees algorithm, a supervised learning method that is based on function approximation by optimizing specific loss functions as well as applying several regularization techniques.


Accuracy = 0.769,
Training error = 0.0,
Misclassification error = 0.231

Mean cross-validation score = 0.75
K-fold CV average score(K=10) = 0.71

After using Early Stopping

GridSearchCV class from the scikit-learn is used to find the best hyperparameters. Grid search performs a sequential search to find the best hyperparameters. It iteratively examines all combinations of the parameters for fitting the model. For each combination of hyperparameters, the model is evaluated using the k-fold cross-validation.
Early_stopping_round=25 from Logloss Graph



Accuracy = 0.783,
Training error = 0.03,
Misclassification error = 0.217

The ROC curve was plotted.
Other evaluation parameters for this model include:

F1 Score = 0.781;
Recall score = 0.760;
AUC = 0.8

The model can be viewed at:
https://drive.google.com/file/d/1fX4XvJvVSEkx9cDs-1RGyg20g856LNS4/view?usp=sharing



8. K-fold cross-validation on KNN
We have defined a dictionary of KNN parameters for the grid search. Here, we consider K values between 3 and 20 and p values of 1 (Manhattan), 2 (Euclidean), and 5 (Minkowski). Second, we passed the KNeighborsClassifier() and KNN_params as the model and the parameter dictionary into the GridSearchCV function. In addition, we include the repeated stratified CV method (cv=cv_method). Also, we tell sklearn which metric to optimize, which is accuracy (scoring='accuracy', refit='accuracy'). Then we fit a KNN model using the training dataset.
After stratified 10-fold cross-validation with 3 repetitions, we observe that the optimal parameters are 17 neighbors using the Euclidean (p=2) distance metric. The mean cross-validation accuracy with the optimal parameters can be extracted using the best_score attribute which equals 0.733.
We define a data frame by combining gs.cv_results_['params'] and gs.cv_results_['mean_test_score']. The gs.cv_results_['params'] is an array of hyperparameter combinations.
We visualize the results using the matplotlib module. The plot shows that K = 17 with the Euclidean distance metric (p=2) outperforms other combinations.

The model can be viewed at:
https://drive.google.com/file/d/1OWBqud9txJOSprlN-RqEbA3F92YQcLTI/view?usp=sharing



9. K-fold cross-validation on Decision Trees


We have fitted a decision tree model and optimized its hyperparameters using a grid search. We performed a grid search over a split criterion, maximum depth, and minimum samples split parameters.
We observe that the best set of hyperparameters is as follows: entropy split criterion with a maximum depth of 7 and min_samples_split value of 3.
Best score = 0.662



10. K-fold cross-validation on Naive Bayes
We fit a Gaussian Naive Bayes model and optimize its only parameter, var_smoothing, using a grid search. Variance smoothing can be considered to be a variant of Laplace smoothing in the sense that the var_smoothing parameter specifies the portion of the largest variance of all features to be added to variances for calculation stability. Gaussian NB assumes that each one of the descriptive features follows a Gaussian, that is, normal distribution. This is highly unlikely in practice, but we can perform what is called a "power transformation" on each feature to make it more or less normally distributed. We perform power transformation using the PowerTransformer method in sklearn. By default, PowerTransformer results in features that have a 0 mean and 1 standard deviation.
Best score = 0.788

The model for Cross-validation on Naive Bayes and Decision trees can be viewed at:
https://drive.google.com/file/d/1CN2AcvcFXUCzBiMOQAFn3k6byI_cZsBk/view?usp=sharing

GOOD MODELS-(misclassification error<0.2)

1. Gaussian Kernel SVM
Accuracy = 0.817,
Training error = 0.04,
Misclassification error = 0.183

2. Logistic Regression
Accuracy = 0.832,
Training error = 0.1,
Misclassification error = 0.168

3. Convolutional Neural Network
Accuracy = 0.811,
Training error = 0.055,
Misclassification error = 0.189


Best Model
Logistic Regression with least misclassification error (0.168) and Training Error for this model = 0.1


ROC curve
A Receiver Operator Characteristic (ROC) curve is a graphical plot used to show the diagnostic ability of binary classifiers. A ROC curve is constructed by plotting the true positive rate (TPR) against the false positive rate (FPR). The true positive rate is the proportion of observations that were correctly predicted to be positive out of all positive observations (TP/(TP + FN)). Similarly, the false positive rate is the proportion of observations that are incorrectly predicted to be positive out of all negative observations (FP/(TN + FP)). The ROC curve shows the trade-off between sensitivity (or TPR) and specificity (1 – FPR). Classifiers that give curves closer to the top-left corner indicate better performance. As a baseline, a random classifier is expected to give points lying along the diagonal (FPR = TPR). The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.
The predictive performance of a classifier can be quantified in terms of the area under the ROC curve (AUC), which lies in the range [0,1]. A random classifier will have an AUC close to 0.5 and an ideal classifier AUC equal to 1.Usually, the AUC is in the range [0.5,1] because useful classifiers should perform better than random. In principle, however, the AUC can also be smaller than 0.5, which indicates that a classifier performs worse than a random classifier.

When we fit a logistic regression model, it can be used to calculate the probability that a given observation has a positive outcome, based on the values of the predictor variables. To determine if an observation should be classified as positive, we can choose a cut-point such that observations with a fitted probability above the cut-point are classified as positive and any observations with a fitted probability below the cut-point are classified as negative.
The ROC curve shows us the values of sensitivity vs. 1-specificity as the value of the cut-off point moves from 0 to 1.  A model with high sensitivity and high specificity will have a ROC curve that hugs the top left corner of the plot. A model with low sensitivity and low specificity will have a curve that is close to the 45-degree diagonal line.
 
The AUC (area under the ROC curve) gives us an idea of how well the model can distinguish between positive and negative outcomes. The higher the AUC, the better the model is at correctly classifying outcomes. In our case, we can see that the AUC is 0.87. We can use AUC to compare the performance of two or more models. The model with the higher AUC is the one that performs best.
 
 

 
 
Here the curve labeled as ROC curve (black) is of Convolutional Neural Network (classifier=Sequential)
 
The ROC curve for all models have been plotted and compared for model performance which can be viewed at:
https://drive.google.com/file/d/1YNapYmQnb9u3rKTOqZNVK67Zo5QzwcQu/view?usp=sharing
 
 

 
The ROC for the Logistic Regression Model:
https://drive.google.com/file/d/1o0KZmMjHWJWzHqRvRmaLMKyj2Be3fJOW/view?usp=sharing

