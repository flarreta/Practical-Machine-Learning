# Practical Machine Learning: final project
### Author: Francisco Larreta

## Background and Instructions
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, we are using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

## Summary
First, after cleaning the data and setting a seed, we are partitioning the training data available in two subsets to evaluate the in-sample error in each of the models built. Next, we are creating two models, one with Decision Tree method and another with Random Forest Method. Finally, we choose the best model observing the lowest in-sample error between both of the models, and using that model to find the predictions for the tresting data available for 20 new observations.

## Required Packages and Setting Seed
```{r message = FALSE}
library(caret)
library(randomForest)
library(rpart)
library(rattle)
set.seed(101010)
```

## Training Data
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

```{r}
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
```

## Test Data
The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

```{r}
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))
```

## Cleaning Data

### Missing Values
First, we are removing all variables that are composed with missing values.
```{r}
training <- training[ ,colSums(is.na(training)) == 0]
testing <- testing[ ,colSums(is.na(testing)) == 0]
dim(training)
dim(testing)
```
 
### Irrelevant Variables
Second, we are removing the first seven columns, as we can observe that are irrelevant for this study.
```{r}
colnames(training)
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
```

## Data Partitioning
Next, for our cross- validations, we select a 70% partition of our initial training data to create the training subset data that we are using for the model, and a testing subset data for our in-sample predictions.
```{r}
train <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
trainset <- training[train,]
testset <- training[-train,]
trainset$classe <- as.factor(trainset$classe)
testset$classe <- as.factor(testset$classe)
```

The final dimensions of the subsets data are the following
```{r}
charac <- cbind(dim(trainset),dim(testset))
colnames(charac) <- c("Training Set", "Testing Set")
rownames(charac) <- c("Observations", "Variables"); charac
```

## ML Algorithms
We are comparing two types of models, Decision Tree and Random Forest, training and testing each model with the subsets built previously.
### Decision Tree
```{r}
mod1 <- rpart(classe ~ ., data = trainset, method = "class")
predict1 <- predict(mod1, testset, type = "class")
cm1 <- confusionMatrix(predict1, as.factor(testset$classe))
```

### Random Forest
```{r}
mod2 <- randomForest(classe ~ ., data = trainset)
predict2 <- predict(mod2, testset, type = "class")
cm2 <- confusionMatrix(predict2, as.factor(testset$classe))
```

### Best Model With In Sample Predictions
Using Confusion Matrixes, we found the following results:
```{r}
results <- as.table(rbind(cbind(cm1$overall["Accuracy"], cm2$overall["Accuracy"]),
                    cbind(1-cm1$overall["Accuracy"], 1-cm2$overall["Accuracy"])))
colnames(results) <- c("Decision Tree Model", "Random Forest Model")
rownames(results)[2] <- "In-sample error"; results
```
We can conclude that there is a better prediction with our Random Forest Model, obtaining an in-sample error of 0.006, which is much lower than an error of 0.257 in our Decision Tree Model. Using the Random Forest Model, we can expect to have almost no error in the predictions with our out-of-sample testing data.

### Out Of Sample Predictions
The predictions that we are using of the submission are obtained with the Random Forest Model as the following:
```{r}
predict3 <- predict(mod2, testing); predict3
```

