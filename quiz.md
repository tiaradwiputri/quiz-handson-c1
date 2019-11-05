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

Our target variable has a balance proportion between resign and not. We do not need to do pre-process for make it balance either using upsampling or downsampling. Next step we will split `turnover` dataset for train data and test data in order to make model fitting and model validation. Split `turnover` dataset for train data with the proportion of data is 80% and store it with `train` object and the rest of it for `test` object to do model validation. Use `set.seed()` with seed 100 and `sample()` to randomize `turnover` dataset before. 

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

```
# your code here
```

3. What can we interpret based on the output above?
  - [ ] Probability if an employee has a work accident being not resigned is 0.23.
  - [ ] Odds ratio employee has work accident being not resigned is about 0.23 more likely than an employee not has a work accident.
  - [ ] Odds ratio employee has work accident being resigned is about 1.44 less likely than an employee not has a work accident.

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

We have learn to make predictive model and try to predict value using `test` in logistic regression model and `test_knn` in k-nearest neighbor model. The last step we must do is checking our model performance to predict the unseen data. In this step, try to make the confusion matrix of model performance in the logistic regression model based on `test` data and `pred_value`. Use the positive value is "1".

```
# your code here

```

Make the same confusion matrix but using `model_knn`.

```
# your code here

```

### Model Evaluation Quiz

Let's say that we are as an Human Resource (HR) that used this machine learning technique to predict employee resign or not. As HR, we want to know which employee has high potential to resign in order to save our cost to find another recruiter. So we want to get a high amount of employee that has the potential to resign based on the historical data. By this metric that we used, we can take precautions so that the employee does not resign.

7. What is the right metric we used to check the performance of our model based on the condition above?"
  - [ ] Recall
  - [ ] Specificity
  - [ ] Accuracy
  - [ ] Precision

8. How much the value based on the metric model performance we used in the logistic regression model?"
  - [ ] 0.7754
  - [ ] 0.8564
  - [ ] 0.8809
  - [ ] 0.7407

9. Which model we want to use as an HR if we want by our model we can learn and make a decision more clearly?
  - [ ] K-nn, because the metric performance is bigger than logistic regression, so we can use the model to predict again. 
  - [ ] Logistic regression, because the metric performance not too worse than k-nn.
  - [ ] Logistic regression, because we can interpret each predictor variable, so we can give decision clearly.
  - [ ] K-nn, because this model more precisious in predict each class target. 
