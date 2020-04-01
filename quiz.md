# Classification 1 Quiz

This quiz is part of Algoritma Academy assessment process. Congratulations on completing the first Classification in Machine Learning course! We will conduct an assessment quiz to test practical classification model techniques you have learned on the course. The quiz is expected to be taken in the classroom, please contact our team of instructors if you missed the chance to take it in class.

To complete this assignment, you are required to build your classification model to classify the characteristics of employees who have resigned and have not. Use Logistic Regression and k-Nearest Neighbor algorithms by following these steps:

# Data Exploration

Let us start by preparing and exploring the data first. In this quiz, you will be using the turnover of employee data (`turnover`). The data is stored as a .csv format in this repository as `turnover_balance.csv` file. Import your data using `read.csv` or `read_csv` and save as `turnover` object. Before building your classification model, you will need to perform an exploratory analysis to understand the data. Glimpse the structure of our `turnover` data! You can choose either `str()` or `glimpse()` function.

```
# your code here
```

Turnover data consists of 10 variables and 7.142 rows. This dataset is a human resource data that shows historical data of employee characteristics who will resign or not. Below is more information about the variable in the dataset:

  - `satisfaction_level`: the level of employee satisfaction working in a company
  - `last_evaluation`: employee satisfaction level at the last evaluation
  - `number_project`: the number of projects the employee has received
  - `average_monthly_hours`: average hours worked per month
  - `time_spend_company`: length of time in the company (years)
  - `work_accident`: presence or absence of work accident, 0 = none, 1 = there
  - `promotion_last_5years`: ever got a promotion in the last 5 years, 0 = no, 1 = yes
  - `division`: name of department or division
  - `salary`: income level, divided into low, medium and high
  - `left`: employee history data resigned, 0 = no, 1 = yes
  
In this quiz, we will try to predict whether or not the employee has a resignation tendency using the `left` column as our target variable. Please change the class of `Work_accident`, `left`, and `promotion_last_5years` column to be in factor class as it should be.

```
# your code here

```

For example, as HR, we are instructed to investigate the division that has a long history of an employee resigning based on average monthly hours. Let's do some aggregation of `average_monthly_hours` for each division. Try to aggregate the average of `average_monthly_hours` from all employees that have left the company and answer the following question.

```
# your code here

```
___
1. Based on the aggregation data that you have analyzed, which division has the highest average of monthly hours?
  - [ ] Marketing division
  - [ ] Technical division
  - [ ] Sales division
  - [ ] Accounting division
___

# Data Preprocessing

After conducting the data exploratory, we will go ahead and perform preprocessing steps before building the classification model. Before we build the model, let us perform dataset splitting for cross validation process. For future reference, please store it as `train` and `test` object.

> **Quiz notes:** Make sure you use `RNGkind()` before splitting to ensure same random seed among different R versions

```
RNGkind(sample.kind = "Rounding")
set.seed(100)
# your code here

```

Now take a look at the proportion of our target variable for both the original dataset, `train` and `test`. Make sure `train` and `test` dataset are distributed with a similar target class proportion.
___
2. Based on the proportions of `train` and `test`, is the distribution of the target class proportion is similar and can be considered as balanced?
  - [ ] The `train` and `test` is similar but not balanced
  - [ ] The `train` and `test` is not similar but not balanced
  - [ ] The `train` and `test` is similar and balanced
  - [ ] The `train` and `test` is not similar and not balanced
___

# Logistic Regression Model Fitting

After we have split our dataset in train and test set, let's try to model our `left` variable using all of the predictor variables to build a logistic regression. Remember, we are not using `turnover` dataset any longer, and we will be using `train` dataset instead.

```
model_logistic <- glm()
```

Based on the `model_logistic` you have made above, take a look at the summary of your model using `summary()` function.

```
# your code here
```
___
3. Logistic regression is one of interpretable model. We can explain how likely each variable are predicted to the class we observed. Based on the model summary above, what can be interpreted from the `Work_accident` coeficient?
  - [ ] The probability of an employee that had a work accident not resigning is 0.21.
  - [ ] Employee who had a work accident is about 0.21 more likely to resign than the employee who has not.  
  - [ ] Employee who had a work accident is about 1.57 less likely to resign than the employee who has not.  
___

# K-Nearest Neighbor Model Fitting

Now let's try to explore the classification model using the k-Nearest Neighbor algorithm. In the k-Nearest Neighbor algorithm, we need to perform one more step of data preprocessing. For both our `train` and `test` set, drop the categorical variable from each column except our `left` variable. Separate the predictor and target in-out `train` and `test` set.

```
# predictor variables in `train`
train_x <-

# predictor variables in `test`
test_x <-

# target variable in `train`
train_y <-

# target variable in `test`
test_y <-
```

Recall that the distance calculation for kNN is heavily dependent upon the measurement scale of the input features. If any variable that have high different range of value could potentially cause problems for our classifier, so let's apply normalization to rescale the features to a standard range of values. Recall you can use the `scale()` function to perform a Z normalization for our variables.

```
# your code here

# scale train_x data
train_x <- __

# scale test_x data
test_x <- __
```

Once you prepared the dataset, you will need to perform model tuning by specifying the **K** parameter for our KNN model. In practice, choosing k depends on the difficulty of the concept to be learned and the
number of records in the training set data.

___
4. The method for getting K value, does not guarantee you to get the best result. But, there is one common practice for determining the number of K. Which method is generally useful to start out picking the best K estimate?
  - [ ] square root by number of row 
  - [ ] number of row
  - [ ] 1
___

Say we decided to use `k=75`, we then proceed to use model our data. Complete the following code to answer the following questions!

```
library(class)
model_knn <- knn(train = ______, test = ________, cl = _______, k = 75)
```

___
5. Fill the missing code here based on the picture above and choose the right code for building the knn model!
  - [ ] model_knn <- knn(train = train_y, test = test_y, cl = test_y, k = 75)
  - [ ] model_knn <- knn(train = train_x, test = test_y, cl = test_x, k = 75)
  - [ ] model_knn <- knn(train = train_x, test = test_x, cl = train_y, k = 75)
  - [ ] model_knn <- knn(train = train_x, test = train_y, cl = train_x, k = 75)
___

# Model Benchmarking

By now you should have two models: `model_knn` and `model_logistic`. You will now compare the performance for both model by predicting the `test` dataset prepared on the earlier section. Recall that the y modelled in logistic regression is the log of odd, make sure you store the probability value for each test set observations: 

```
prob_value <- ___
```
___
6. Using the threshold of 0.5, how many predictions do our `model_logistic` predict for each class?
  - [ ] class 0 = 714, class 1 = 715
  - [ ] class 0 = 524, class 1 = 905
  - [ ] class 0 = 590, class 1 = 839
 ___ 

Since we have now acquired both the predicted result of `model_knn` and `model_logistic`, we can perform a cross validation using `test` set true label to assess wether or not both model did a good job in predicting unseen data. In this step, try to make the confusion matrix for both class and compare the models performance.

```
# your code here

```

Let's say that we are working as an HR staff in a company and are utilizing this model to predict the probability of an employee resigning. Say we would want to know which employee has a high potential of resigning so that we can take a precautionary approach as soon as possible in order for us to mitigate and reduce the number of resigning employees as much as possible. Based on the stated case example try to answer the following questions!

___
7. Which one is the right metric for us to evaluate the numbers of resigning employees that we can detect?
  - [ ] Recall
  - [ ] Specificity  
  - [ ] Accuracy  
  - [ ] Precision  

___
8. Using the metrics of your answer in the previous question, which of the two models has a better performance in detecting resigning employees?
  - [ ] Logistic Regression
  - [ ] K-Nearest Neighbor  
  - [ ] Both has more or less similar performance  

___
9.  Which one is more suitable to use if we would like to gain insight on how each predictor variables is affecting the resign decisions for the resigned employees?
  - [ ] K-NN, because it tends to have a higher performance than logistic regression
  - [ ] Logistic regression, because it has a lower performance than K-NN
  - [ ] Logistic regression, because each coefficient can be transformed into an odds ratio
  - [ ] K-NN, because it results in a better precision score for the positive class
___
