# loading libraries
rm(list=ls())
library(xgboost)
library(data.table)

#Reading the data
setwd("..")

train <- fread("datahulk/train.csv")
test <- fread("datahulk/test.csv")
train[, train_flag := 1]
test[, Outcome := 0]
test[, train_flag := 0]

#Combining Train and test data
X_panel <- rbind(train, test)

X_train <- X_panel[train_flag == 1]
X_test <- X_panel[train_flag == 0]

X_features <- c("Volume" , "Three_Day_Moving_Average", "Five_Day_Moving_Average" ,"Ten_Day_Moving_Average",    "Twenty_Day_Moving_Average" ,"True_Range", "Average_True_Range" , "Positive_Directional_Movement", "Negative_Directional_Movement" )

X_target <- X_train$Outcome
X_train1 = X_train[, X_features, with = FALSE]

xgtrain <- xgb.DMatrix(data = as.matrix(X_train[, X_features, with = FALSE]), label = X_target, missing = NA)
xgtest <- xgb.DMatrix(data = as.matrix(X_test[, X_features, with = FALSE]), missing = NA)

Y_train = X_target
X_test1 = X_test[, X_features, with = FALSE]

best_logloss = Inf
best_logloss_index = 0

for (iter in 1:50) {
  print (iter)
  param <- list(objective = "binary:logistic",
                eval_metric = "logloss",
                max_depth = sample(4:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .9), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=as.matrix(X_train1), label=as.matrix(Y_train), params = param, nthread=6,  nfold = cv.nfold, missing = "NAN", nrounds=cv.nround, verbose = T, early_stopping_rounds = 20, maximize = F)
  
  min_logloss = min(mdcv$evaluation_log$test_logloss_mean)
  min_logloss_index = which.min(mdcv$evaluation_log$test_logloss_mean)
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}

# params <- list()
# params$objective <- "binary:logistic"
# params$eta <- 0.1
# params$max_depth <- 5
# params$subsample <- 0.9
# params$colsample_bytree <- 0.9
# params$min_child_weight <- 2
# params$eval_metric <- "logloss"

nround = best_logloss_index
set.seed(best_seednumber)
model_xgb1 <- xgb.train(xgtrain, params=best_param, missing = "NAN" , nrounds=nround, verbose = T, nthread=6)


## submission
pred <- predict(model_xgb1, xgtest)

submit <- data.table(ID = X_test$ID, Outcome = pred)
write.csv(submit, "xgb2.csv", row.names = FALSE)
