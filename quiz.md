# 1 Data Exploration

In this quiz, we will be using turnover of employee data (`turnover`). The data is stored as csv file in this repository as `turnover_balance.csv` file. Before you build your classification model, you need to perform an exploratory analysis to be able to understand the data. Glimpse the structure of our `turnover` data! You can choose either `str()` or `glimpse()` function.

```
# your code here

```

Turnover data is consist of 10 variables and 14.999 row. Turnover dataset is a human resource data that shows historical data of employee characteristics that are resigned or not. This is more information about variable in the dataset:

  - `satisfaction_level`: the level of employee satisfaction working in a company
  - `last_evaluation`: employee satisfaction level at the last evaluation
  - `number_project`: the number of projects the employee has received
  - `average_monthly_hours`: average hours worked per month
  - `time_spend_company`: length of time in the company (years)
  - `work_accident`: presence or absence of work accident, 0 = none, 1 = there
  - `left`: employee history data resigned, 0 = no, 1 = yes
  - `promotion_last_5years`: ever got a promotion in the last 5 years, 0 = no, 1 = yes
  - `sales`: name of department or division
  - `salary`: income level, divided into low, medium and high
  
In this quiz, we will try to predict wether or not the employee has resignation tendency using `left` column as our target variable. Let's take a look mean of `average_monthly_hours` of each division and history of employee resign.

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

We will move to pre-process step before build classification model using `turnover` dataset. Let's take a look the proportion of our class target in `left` column before building the model. 

```
# your code here

```

Our target variable has a balance proportion between both classes. So our next step is to split the dataset into a train and test set in order to perform a model validation. Split `turnover` dataset into a train data using 80% of the data and store it under `train` object. Use the rest 20% of the data as the test set and store it under `test` object. Use `set.seed()` with seed 100 and `sample()` to randomize `turnover` dataset before. 

> **Notes:** Make sure your R version is 3.6 

```
# RNGkind(sample.kind="Rounding")
# set.seed(100)
# your code here

```

## Data Pre-Process Quiz

Let's take a look distribution of proportion in `train` and `test` data, and try to answer the question below. Please rounding the proportion using two decimal number.

```
# your code here

```

2. Based on proportion of `train` and `test` above, is the distribution of each class is a balance? Why the distribution of each class must balance?
  - [ ] No, it is not.
  - [ ] Yes, it is. Distribution of each class does not need to be balance. 
  - [ ] No, it is not. Distribution of each class needs to be balanced to make the model not miss classify each class. 
  - [ ] Yes, it is balance. Distribution of each class needs to be balanced to make model learn both in each class as well. 

# 3.1 Logistic Regression Model Fitting

We have a `train` and `test` dataset. Let's try to model the `left` variable with all of the predictor variables using the logistic regression model. Please store your model in `model_logistic`. Remember, we have not using `turnover` dataset any longer, but using `train` dataset.

```
# model_logistic <- 

```

Based on the `model_logictic` you have made above, make the summary of the model.

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

# 3.2 Model Fitting K-Nearest Neighbor Section

## Model Fitting Tutorial

In k-Nearest Neighbor algorithm, we need more data pre-process before make a modeling, because in k-Nearest Neigbor we must set the right **'k'** so our model can predict the target variable well. Use the `train` dataset, drop the factor variables except the target variable `left` then scale the numeric column and store it in `train_knn`. Use the `test` dataset and drop the factor variables except the target variable `left` then scale the numeric column using attribute center and scale from `train_knn` data and store it in`test_knn`. 

```
# your code here

```

After we have done in additional pre-processing step, we move to build k-NN model. But, do not forget to find the right k first. To get the right k, please use the `train_knn` datasets information if needed. If you got decimal number, do not forget to round it for getting odd values. 

```
# your code here

```

## K-Nearest Neighbor Quiz

Using k value we have calculate in the section before, try to modeling the `train_knn` data to make knn model. Store the model in `model_knn`. 

```
# your code here

```

The method to acquire K value, however, does not guarantee you to acquire the best result. There are some other way to try out different K values.

4. What method we can use to choose an appropriate k?
  - [ ] square root by number of row 
  - [ ] number of row
  - [ ] use k = 1


If you have succeeded in making the knn model, try to answer this question.

![](model.png)

5. Fill the missing code here based on the picture above, and choose the right code for build knn model!
  - [ ] model_knn <- knn(train = train_knn, test = test_knn[,-8], cl = train_knn[,-8], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,8], test = test_knn[,-8], cl = train_knn[,8], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,-8], test = test_knn[,-8], cl = train_knn[,8], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,-8], test = test_knn[,-8], cl = train_knn[,-8], k = 75)

# 4 Prediction

Now let's get back to our `model_logistic`. In this section, try to predict `test` data using `model_logistic` return the probability value and store it under `pred_value` object.

```
# your code here

```

Using `pred_value`, we have got the probability of employee getting resign or not. Use threshold 0.45 to classify if the probability of employee getting resigns or not more than 0.45, make it 1, if it is not, make it 0. 

```
# your code here

```

## Prediction Quiz

Based on the prediction value above, try to answer the following question.

6. How many prediction does our model generate for each class?
  - [ ] class 0 = 714, class 1 = 715
  - [ ] class 0 = 524, class 1 = 905
  - [ ] class 0 = 592, class 1 = 837
  

# 5 Model Evaluation

In the previous sections, we have performed a prediction using both Logistic Regression and K-NN algorithm. However, we need to validate wether or not our model did a good job in predicting unseen data. In this step, try to make the confusion matrix of model performance in logistic regression model based on `test` data and `pred_value` and use the positive class is "1".

```
# your code here

```

Make the same confusion matrix but using `model_knn`.

```
# your code here

```

### Model Evaluation Quiz

Let's say that we worked as an HR staff in a company and is utilizing this model to predict the probability of an employee resigning. As an HR, we would want to know which employee has a high potential of resigning so that we are able to take a precaution approach as soon as possible. Now try to answer the following questions.

7. Which one is a good metrics for us to evaluate numbers ofresigning employee that we are able to detect early?
  - [ ] Recall  
  - [ ] Specificity  
  - [ ] Accuracy  
  - [ ] Precision  

8. Using the metrics of your answer in the previous question, which of the 2 model has a better performance in detecting resigning employees?  
  - [ ] Logistic Regression  
  - [ ] K-Nearest Neighbor  
  - [ ] Both has more or less similar performance  
  
9.  Which model we want to use as an HR if we want to make a decision more clearly?
  - [ ] K-nn, because the model give higher performance than logistic regression
  - [ ] Logistic regression, because the model less similar performance with k-nn
  - [ ] Logistic regression, because we can interpret each predictor variable
  - [ ] K-nn, because this model more precisious in predict each class target
