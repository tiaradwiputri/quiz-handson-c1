Congratulations! This is the end of Classification in Machine Learning I unit. The last part of this course is closed by filling this quiz.

To complete this assignment, you need to build your classification model to classify the characteristics of employee being resign or not using Logistic Regression and k-Nearest Neighbor algorithms by following these steps:

# 1 Data Exploration

Let us start by preparing and exploring the data first. In this quiz, you will be using turnover of employee data (`turnover`). The data is stored as a .csv format in this repository as `turnover_balance.csv` file. Before building your classification model, you need to perform an exploratory analysis to understand about the data. Glimpse the structure of our `turnover` data! You can choose either `str()` or `glimpse()` function.

```
# your code here
```

Turnover data consist of 10 variables and 7.142 row. This dataset is a human resource data that shows historical data of employee characteristics who will resigned or not. This is more information about variable in the dataset:

  - `satisfaction_level`: the level of employee satisfaction working in a company
  - `last_evaluation`: employee satisfaction level at the last evaluation
  - `number_project`: the number of projects the employee has received
  - `average_monthly_hours`: average hours worked per month
  - `time_spend_company`: length of time in the company (years)
  - `work_accident`: presence or absence of work accident, 0 = none, 1 = there
  - `left`: employee history data resigned, 0 = no, 1 = yes
  - `promotion_last_5years`: ever got a promotion in the last 5 years, 0 = no, 1 = yes
  - `division`: name of department or division
  - `salary`: income level, divided into low, medium and high
  
In this quiz, we will try to predict whether or not the employee has a resignation tendency using `left` column as our target variable. Please change the class of `Work_accident`, `left`, and`promotion_last_5years` column to be in factor class as it should be.

Let's say, as HR we are instructed to investigate the division that has a high history of an employee resigning based on average monthly hours we got. Let's do some aggregation of `average_monthly_hours` of each division and history of employee resign to get the answer. You can use `group_by()` function by `division` and `left` variable and `summarise()` the mean of `average_monthly_hours` variable and arrange it by ascending of the mean value of `average_monthly_hours` using `arrange()` function.

```
# your code here
```

## Data Exploration Quiz

1. Based on the aggregation data you have analyzed, which division has the highest average of monthly hours with a high probability of resigning?
  - [ ] Marketing division
  - [ ] Technical division
  - [ ] Sales division
  - [ ] Accounting division

# 2 Data Pre-Processing

After conducting the data exploratory, we will go ahead and perform pre-processing steps before building the classification model. Before we build the model, let us take a look the proportion of our target variable in the `left` column using `prop.table(table(data))` function.

```
# your code here
```

It seems like our target variable has a balance proportion between both classes. Before we build model, we should split the dataset into train and test data in order to perform model validation. Split `turnover` dataset into 80% train and 20% test proportion using `sample()` function and use `set.seed()` with the seed 100. Store it as `train` and `test` object.

> **Notes:** Make sure your R version is 3.6 or above

```
set.seed(100)
# your code here
```

## Data Pre-Process Quiz

Let's take a look distribution of proportion in `train` and `test` data using `prop.table(table(data))` fuction to make sure in train and test data has balance or not distribution of each class target. Please round the proportion using two decimal numbers using `round()` function.

```
# your code here

```

2. Based on the proportion of `train` and `test`, is the distribution of each class can be considered as balance? Why do we need to make sure that each class has a balance proportion for each class?
  - [ ] No, it is not.
  - [ ] Yes, it is, but it is not necessary to balance the class proportion.  
  - [ ] No, it is not. The distribution of each class needs to be balanced to prevent any misclassified observation.  
  - [ ] Yes, it is. The distribution of each class needs to be balanced so that the model can learn the characteristics for each class equally.

# 3.1 Logistic Regression Model Fitting

After we have splitted our dataset in train and test set, let's try to model our `left` variable using all of the predictor variables to build a logistic regression. Please use the `glm(formula, data, family = "binomial")` to do that and store your model under `model_logistic` object. Remember, we have not using `turnover` dataset any longer, and we will be using `train` dataset instead.

```
# model_logistic <- glm()
```

Based on the `model_logictic` you have made above, take a look at the summary of your model using `summary()` function.

```
# your code here
```

## Logistic Regression Quiz

Based on the model summary above, try to answer the following question.

3. What can be interpreted from the `Work_accident` variable based on the output above?
  - [ ] The probability of an employee that had a work accident not resigning is 0.23.
  - [ ] Employee that had a work accident is about 0.23 more likely to resign than the employee that has not.  
  - [ ] Employee that had a work accident is about 1.44 less likely to resign than the employee that has not.  

# 3.2 K-Nearest Neighbor Model Fitting

Now let's try to explore classification model using k-Nearest Neighbor algorithm. In k-Nearest Neighbor algorithm, we need to perform one more step of data preprocessing. For both our `train` and `test` set, drop the categorical variable from each column except our `left` variable. Separate the predictor and target in-out `train` dan `test` set.

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

After we separate the target and predictor variables, in `train_x`, please scale each column using `scale()` function. In `test_x`, please scale each column using attribute center and scale of `train_x`. Please use `scale(data_test, center = attr(data_train, "scaled:center"), scale = attr(data_train, "scaled: scale"))` to scale the `test_x` data.

```
# scale train_x data
train_x <- scale()

# scale test_x data
test_x <- scale()
```

After we have done performing data scaling, we need to find the right **K** to use for our K-NN model. To get the right K, please use the number of row from our `train_x` datasets. If you have got decimal number, do not forget to round it and make sure you end up with an odd number to prevent voting tie break.

```
# your code here
```

## K-Nearest Neighbor Quiz

The method to acquire K value, however, does not guarantee you to acquire the best result. There are some other way to try out different K values.

4. What method we can use to choose an appropriate k?
  - [ ] square root by number of row 
  - [ ] number of row
  - [ ] use k = 1

Using K value we have calculated in the section before, try to predict `test_y` using `train_x` dan `train_y` dataset. To make the k-nn model, please use the `knn()` function and store the model under `model_knn` object.

```
model_knn <- knn()
```

Next, take a look at the following syntax:

```
library(class)
model_knn <- knn(train = ______, test = test_knn[,-6], cl = _______, k = 75) 
```

5. Fill the missing code here based on the picture above and choose the right code for building the knn model!
  - [ ] model_knn <- knn(train = train_knn, test = test_knn[,-6], cl = train_knn[,-6], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,6], test = test_knn[,-6], cl = train_knn[,6], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,-6], test = test_knn[,-6], cl = train_knn[,6], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,6], test = test_knn[,-6], cl = train_knn[,-6], k = 75)

# 4 Prediction

Now let's get back to our `model_logistic`. In this section, try to predict `test` data using `model_logistic` return the probability value using `predict()` function with `type = "response"` in the parameter function and store it under `prob_value` object.

```
prob_value <-
```

Because the result of the prediction in the logistic model is probability, then we have to change it into class form according to the target class we have. Now, given a threshold of 0.45, try to classify whether or not an employee can be predicted to resign. Please use `ifelse()` function and store the prediction result under `pred_value` object.

```
pred_value <-
```

## Prediction Quiz

Based on the prediction value above, try to answer the following question.

6. How many prediction does our model_logistic generate for each class?
  - [ ] class 0 = 714, class 1 = 715
  - [ ] class 0 = 524, class 1 = 905
  - [ ] class 0 = 592, class 1 = 837
  

# 5 Model Evaluation

In the previous sections, we have performed a prediction using both Logistic Regression and K-NN algorithm. However, we need to validate whether or not our model did a good job of predicting unseen data. In this step, try to make the confusion matrix of model performance in the logistic regression model based on `test` data and `pred_value` and use the positive class is "1".

**Note:** do not forget to do the explicit coercion `as.factor()`.

```
# your code here
```

Make the same confusion matrix for `model_knn` prediction result of `test_y`.

```
# your code here
```

### Model Evaluation Quiz

Let's say that we are working as an HR staff in a company and is utilizing this model to predict the probability of an employee resigning. As an HR, we would want to know which employee has a high potential of resigning so that we can take a precaution approach as soon as possible. Now try to answer the following questions.

1. Which one is the right metrics for us to evaluate the numbers of resigning employee that we can detect?
  - [ ] Recall
  - [ ] Specificity  
  - [ ] Accuracy  
  - [ ] Precision  

8. Using the metrics of your answer in the previous question, which of the two model has a better performance in detecting resigning employees?
  - [ ] Logistic Regression
  - [ ] K-Nearest Neighbor  
  - [ ] Both has more or less similar performance  
  
9.  Now recall how we have learned the advantage of each model. Which one are more suitable to use if we aimed for model interpretability?
  - [ ] K-NN, because it tends to have a higher performance than logistic regression
  - [ ] Logistic regression, because it has a lower performance than K-nn
  - [ ] Logistic regression, because each coefficient can be transformed into odds ratio
  - [ ] K-NN, because it results in a better precision score for the positive class
