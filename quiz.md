# 1 Data Exploration

In this quiz, we will be using turnover of employee data (`turnover`). The data is stored as csv file in this repository as `turnover_balance.csv` file. Before you build your classification model, you need to perform an exploratory analysis to be able to understand the data. Glimpse the structure of our `turnover` data! You can choose either `str()` or `glimpse()` function.

```
# your code here

```

Turnover data is consist of 10 variables and 7.142 row. Turnover dataset is a human resource data that shows historical data of employee characteristics that are resigned or not. This is more information about variable in the dataset:

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
  
In this quiz, we will try to predict wether or not the employee has resignation tendency using `left` column as our target variable. Please change class of `Work_accident`, `left`, and`promotion_last_5years` column to be in factor class. Let's take a look mean of `average_monthly_hours` of each division and history of employee resign.

```
# your code here

```

## Data Exploration Quiz

1. Based on aggregation data you have created, which division has the highest average of monthly hours with high probability of resigning?
  - [ ] IT division
  - [ ] Technical division
  - [ ] Sales division
  - [ ] Accounting division

# 2 Data Pre-Processing

After we are done with data exploratory, we will go ahead and perform pre-processing steps before building the classification model. Before we build the model, let's take a look the proportion of our target variable in `left` column. 
```
# your code here

```

Our target variable has a balance proportion between both classes. So our next step is to split the dataset into a train and test set in order to perform a model validation. Split `turnover` dataset into a train data using 80% of the data and store it under `train` object. Use the rest 20% of the data as the test set and store it under `test` object.

> **Notes:** Make sure your R version is 3.6 or above

```
set.seed(100)
# your code here

```

## Data Pre-Process Quiz

Let's take a look distribution of proportion in `train` and `test` data, and try to answer the question below. Please rounding the proportion using two decimal number.

```
# your code here

```

2. Based on proportion of `train` and `test`, is the distribution of each class can be considered as balance? Why do we need to make sure that each class has a balance proportion for each class?
  - [ ] No, it is not balance.  
  - [ ] Yes, it is balance, but it is not necessary to balanced between the class proportion.  
  - [ ] No, it is not. Distribution of each class need to be balance to prevent any missclassified observation.  
  - [ ] Yes, it is balance. Distribution of each class need to be balance so that the model can learn the characteristics for each class equally.  

# 3.1 Logistic Regression Model Fitting

After splitting our dataset in train and test set, let's try to model our `left` variable using all of the predictor variables to build a logistic regression. Please store your model in `model_logistic`. Remember, we've not using `turnover` dataset any longer and we will be using `train` dataset instead.

```
# model_logistic <- 

```

Based on the `model_logictic` you have made above, make take a look at the summary of your model.

```
# your code here

```

## Logistic Regression Quiz

Based on the model summary above, try to answer the following question.

3. What can be interpreted from `Work_accident` variable based on the output above?
  - [ ] Probability of an employee that had a work accident not resigning is 0.23.  
  - [ ] Employee that had a work accident is about 0.23 more likely to resign than the employee that has not.  
  - [ ] Employee that had a work accident is about 1.44 less likely to resign than the employee that has not.  

# 3.2 K-Nearest Neighbor Model Fitting

In k-Nearest Neighbor algorithm, we need to perform one more step of data preprocessing. For both our `train` and `test` set, drop the categorical variable from each column except our `left` variable, then scale the numeric column and store it under `train_knn` and `test_knn`.

```
# your code here

```

After we have done performing data scaling, we will need to find the right K to use for our K-NN model. To get the right K, please use the number of row from our `train_knn` datasets. If you've got decimal number, don't forget to round it and make sure you end up with an odd number to prevent voting tie break. 

```
# your code here

```

## K-Nearest Neighbor Quiz

The method to acquire K value, however, does not guarantee you to acquire the best result. There are some other way to try out different K values.

4. What method we can use to choose an appropriate k?
  - [ ] square root by number of row 
  - [ ] number of row
  - [ ] use k = 1

Using K value we've calculate in the section before, try to predict `test_knn` using `train_knn` dataset. 

```
# your code here

```

Next, take a look at the following syntax:

```
library(class)
model_knn <- knn(train = ______, test = test_knn[,-6], cl = _______, k = 75) 
```

5. Fill the missing code here based on the picture above, and choose the right code for build knn model!
  - [ ] model_knn <- knn(train = train_knn, test = test_knn[,-6], cl = train_knn[,-6], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,6], test = test_knn[,-6], cl = train_knn[,6], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,-6], test = test_knn[,-6], cl = train_knn[,6], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,6], test = test_knn[,-6], cl = train_knn[,-6], k = 75)

# 4 Prediction

Now let's get back to our `model_logistic`. In this section, try to predict `test` data using `model_logistic` return the probability value and store it under `pred_value` object.

```
# your code here

```

Now, given a threshold of 0.45, try to classify wether or not an employee can be predicted to resign.

```
# your code here

```

## Prediction Quiz

Based on the prediction value above, try to answer the following question.

6. How many prediction does our model_logistic generate for each class?
  - [ ] class 0 = 714, class 1 = 715
  - [ ] class 0 = 524, class 1 = 905
  - [ ] class 0 = 592, class 1 = 837
  

# 5 Model Evaluation

In the previous sections, we have performed a prediction using both Logistic Regression and K-NN algorithm. However, we need to validate wether or not our model did a good job in predicting unseen data. In this step, try to make the confusion matrix of model performance in logistic regression model based on `test` data and `pred_value` and use the positive class is "1".

```
# your code here

```

Make the same confusion matrix but using `model_knn` prediction result.

```
# your code here

```

### Model Evaluation Quiz

Let's say that we are working as an HR staff in a company and is utilizing this model to predict the probability of an employee resigning. As an HR, we would want to know which employee has a high potential of resigning so that we are able to take a precaution approach as soon as possible. Now try to answer the following questions.

7. Which one is a good metrics for us to evaluate the numbers of resigning employee that we are able to detect?
  - [ ] Recall  
  - [ ] Specificity  
  - [ ] Accuracy  
  - [ ] Precision  

8. Using the metrics of your answer in the previous question, which of the 2 model has a better performance in detecting resigning employees?  
  - [ ] Logistic Regression  
  - [ ] K-Nearest Neighbor  
  - [ ] Both has more or less similar performance  
  
9.  Now recall how we have learned the advantage of each model. Which one are more suitable to use if we aimed for model interpretability?
  - [ ] K-nn, because it tends to have a higher performance than logistic regression
  - [ ] Logistic regression, because it has a lower performance than K-nn
  - [ ] Logistic regression, because each coefficient can be transformed into odds ratio
  - [ ] K-nn, because it results in a better precision score for the positive class
