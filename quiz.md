# 1 Data Exploration Section

## Data Exploration Tutorial

In this quiz, we will be using turnover of employee data (`turnover`). The data is available in your R environment. Before you perform classification, you need to explore the data to get information about the data. Glimpse the data from `turnover` data! You can choose `str()` or `glimpse()`.

```
# your code here

```

As we can see on turnover data, this data consist of 10 variables and 14.999 row. Turnover dataset is a human resource data that shows historical data of employee characteristics that are resigned or not. This is more information about variable in the dataset:

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
  
In this quiz, we will try to predict employee that given a chance to resign or not based on the `left` column as our target variable. Let's take a look mean of `average_monthly_hours` of each division and history of employee resign.

```
# your code here

```

## Data Exploration Quiz


1. Based on aggregation data below, what division that has high average of monthly hours and high potential to resign?
  - [ ] IT division
  - [ ] Technical division
  - [ ] Sales division
  - [ ] Accounting division

# 2 Data Pre-Process Section

## Data Pre-Process Tutorial

After we've done in exploratory data, we move to pre-process step before build classification model using `turnover` dataset. Before we build the model, let's take a look the proportion of our class target in `left` column. 

```
# your code here

```

Our target variable has a balance proportion between resign and not. So we don't need to do pre-process to make it balance either using upsampling or downsampling. So our next step is split `turnover` dataset to be train data and test data in order to make model fitting and model validation. Split `turnover` dataset for train data with the proportion of data is 80% and store it with `train` object and the last one store it with `test` object for our data testing. Use `set.seed()` with seed 100 and `sample()` to randomize `turnover` dataset before split it. 

```
# RNGkind(sample.kind="Rounding")
# set.seed(100)
# your code here

```

## Data Pre-Process Quiz

After splitting dataset, so let's take a look distribution of proportion in `train` and `test` data, and try to answer the question below. Please rounding the proportion using two decimal number.

```
# your code here

```

2. Based on proportion of `train` and `test`, is the distribution of each class is balance? Why the distribution of each class must balance?
  - [ ] No, it is not balance.
  - [ ] Yes, it is balance. Distribution of each class doesn't need to be balance. 
  - [ ] No, it is not balance. Distribution of each class need to be balance to make model not miss classify each class. 
  - [ ] Yes, it is balance. Distribution of each class need to be balance to make model learn both in each class as well. 


# 3.1 Model Fitting Logistic Regression Section

## Model Fitting Tutorial

After cross validation section, we've made `train` and `test` dataset. Let's try to modeling the `left` variable using all of predictor variable using logistic regression model. Please store your model in `model_logistic`. Remember, we've not using `turnover` dataset any longer, but we'll try to make those model using `train` dataset.

```
# model_logistic <- 

```

Based on the `model_logictic` you've made above, make the summary of the model.

```
# your code here

```

## Model Fitting Quiz

On the summary model above, try to interpret one of predictor variable. Let say you pick the `Work_accident` variable.

```
# your code here
```

3. What can we interpret based on the output above?
  - [ ] Probability if employee has work accident being not resign is 0.23.
  - [ ] Odds ratio employee has work accident being not resign is about 0.23 more likely than employee not has work accident.
  - [ ] Odds ratio employee has work accident being not resign is about 1.44 less likely than employee not has work accident.


# 3.2 Model Fitting K-Nearest Neighbor Section

## Model Fitting Tutorial

In k-Nearest Neighbor algorithm, we need more data pre-process before make a modeling. Because in k-Nearest Neigbor we must set the right **'k'** so our model can predict the target variable well. Using `train` and `test` dataset, drop the factor variable exept the target variable `left` then scale the numeric column and store it in `train_knn` and `test_knn`. 

```
# your code here

```

After we have done in additional pre-processing step, we move to build k-NN model. But, don't forget to find the right k first. To get the right k, please use the `train_knn` datasets information if needed. If you've got decimal number, don't forget to round it for getting odd values. 

```
# your code here

```

## Model Fitting Quiz

Using k value we've calculate in the section before, try to modeling the `train_knn` data to make knn model. Store the model in `model_knn`. 

```
# your code here

```

Based on the k value we have acquired, try to answer the following question.

4. What method we can use to choose an appropriate k?
  - [ ] square root by number of row 
  - [ ] elbow method
  - [ ] number of row
  - [ ] use k = 1


If you've success for making the knn model, try to answer this question.

![](model.png)

5. Fill the missing code here based on the picture above, and choose the right code for build knn model!
  - [ ] model_knn <- knn(train = train_knn, test = test_knn[,-8], cl = train_knn[,-8], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,8], test = test_knn[,-8], cl = train_knn[,8], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,-8], test = test_knn[,-8], cl = train_knn[,8], k = 75)
  - [ ] model_knn <- knn(train = train_knn[,-8], test = test_knn[,-8], cl = train_knn[,-8], k = 75)

# 4 Prediction Section

## Prediction Tutorial

Based on the logistic regression model that we've made and store it as `model_logistic`, we want to predict `test` data using those model. In prediction step of logistic regression, we can get log of odds of predict data or probability of predict data. In this section, try to predict `test` data using `model_logistic` return the probability value and store it in `pred_value` object.

```
# your code here

```

Using `pred_value`, we've got the probability employee getting resign or not. Use threshold 0.45 to classify if the probability of employee getting resign or not more than 0.45, make it 1, if its not, make it 0. 

```
# your code here

```

## Prediction Quiz

Based on the prediction value above, try to answer this question.

6. How much our model can predict class 0 and class 1 in logistic regression model?
  - [ ] class 0 = 714, class 1 = 715
  - [ ] class 0 = 524, class 1 = 905
  - [ ] class 0 = 592, class 1 = 837
  

# 5 Model Evaluation Section

## Model Evaluation Tutorial

We've learn to make predictive model and try to predict value using `test` in logistic regression model and `test_knn` in k-nearest neighbor model. Last step we must done is checking our model performance in predict the unseen data. In this step, try to make the confusion matrix of model performance in logistic regression model based on `test` data and `pred_value`. Use the positive value is "1".

```
# your code here

```

Make the same confusion matrix but using `model_knn` using `test_knn` data. Use the positive value is "1". 

```
# your code here

```

### Model Evaluation Quiz

Let's say that we're Human Resource (HR) that used this machine learning technique to predict employee resign or not. As HR, we want to know which employee that has high potential to resign in order to save our cost to find another recruiter. So we want to get a high amount of employee  that has potential to resign based on the historical data. By this metric that we used, we can take precautions so that the employee does not resign.

7. What is the right metric we used to check performance of our model based on the condition above?"
  - [ ] Recall
  - [ ] Specificity
  - [ ] Accuracy
  - [ ] Precision

8. How much the value based on the metric model performance we used in the logistic regression model?"
  - [ ] 0.7754
  - [ ] 0.8564
  - [ ] 0.9033
  - [ ] 0.7407

# 6 Conclusion Section

## Conclusion Tutorial

In the section before, we've try to make model and predict using unseen data. We've try to evaluate our model both of logistic regression and k-nn. Now we want to evaluate which model is giving the best performance to predict `left`. Let's we print both of confusion matrix in logistic model and k-nn model.

```
# your code here

```

```
# your code here

```

## Conclusion Quiz

Based on metric model performance that we used before, try to answer this question. 

9. Which model we want to use as HR if we want by our model we can learn and make decision more clearly?
  - [ ] K-nn, because the metric performance is bigger than logistic regression, so we can use the model to predict again. 
  - [ ] Logistic regression, because the metric performance not too worse than k-nn.
  - [ ] Logistic regression, because we can interpret each predictor variable, so we can give decision clearly.
  - [ ] K-nn, because this model more precisious in predict each class target. 
