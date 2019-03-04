#load libraries
require(caret)
require(dplyr)
library(ROCR)
rm(list=ls())
setwd("../LoanPrediction")
train.data = read.csv("LoanPrediction/train.csv", stringsAsFactors = F, na.strings = "")
test.data = read.csv("LoanPrediction/test.csv", stringsAsFactors = F, na.strings = "")

#DE
test.data$Loan_Status = sample(c('Y','N'), nrow(test.data), replace = T)
data = rbind(train.data, test.data)
sapply(data, function(x){length(unique(x))})
sapply(data, function(x){sum(is.na(x))})

#Changing chracter to factor variable
# for(i in names(data))
# {
#   if(class(data[[i]]) == "character")
#     data[[i]] = as.factor(data[[i]])
# }

# missing values
library(VIM)
mice_plot <- aggr(data[,-c(1,5,7,8,12,13)], col = c('navyblue','yellow'), numbers = TRUE, sortVars = TRUE, labels = names(data[,-c(1,5,7,8,12,13)]), cex.axis = 0.7, gap = 3, ylab = c("Missing data", "Pattern"))

data$Gender[is.na(data$Gender)] = 'unknown'
data$Married[is.na(data$Married)] = 'unknown'
data$Dependents[is.na(data$Dependents)] = 'unknown'
data$Self_Employed[is.na(data$Self_Employed)] = 'unknown'
data$Credit_History[is.na(data$Credit_History)] = 'unknown'

data$LoanAmount[is.na(data$LoanAmount)] = mean(data$LoanAmount, na.rm = T)
data$Loan_Amount_Term[is.na(data$Loan_Amount_Term)] = mean(data$Loan_Amount_Term, na.rm = T)

#Changing chracter to factor variable
for(i in names(data))
{
  if(class(data[[i]]) == "character")
    data[[i]] = as.factor(data[[i]])
}

# univariate analysis
for(i in names(data))
{
  if(class(data[[i]]) %in% c("numeric", "integer"))
    {print (i)
    hist(data[[i]], main = i, xlab = i, breaks = length(data[[i]])/30) 
    plot(data[[i]], main = i, xlab = i) }
}

#sapply(data, function(x){length(unique(x))})

#Separating train and test data set
train.data1 = data[1:nrow(train.data),]
test.data1 = data[ (nrow(train.data)+1):nrow(data), ]
set.seed(1)
train = sample(1:nrow(train.data1), 0.7*nrow(train.data1), replace = F)
dev.data = train.data1[train,]
valid.data = train.data1[-train,]

#Visulaization
for(i in names(dev.data[,-13]))
{
  if(class(dev.data[[i]]) == "factor")
  {
    print(i)
    print(ggplot(dev.data, aes(Loan_Status, ..count.. ,fill = dev.data[[i]])) + geom_bar(aes(), position = "dodge") + labs(x = i, y = "Loan Count"))
  } 
  else
    print( ggplot(dev.data, aes( x = Loan_Status, y = dev.data[[i]] ) ) + stat_summary(fun.y= mean, geom="bar", fill = "red") + labs(y = i, x = "Loan Status"))
}



#Model Building
library(h2o)
localh2o = h2o.init(nthreads = -1)
h2o.clusterIsUp()
train.h2o <- as.h2o(dev.data)
valid.h2o <- as.h2o(valid.data)
colnames(train.h2o)
colnames(valid.h2o)

#dependent variable (Footfall)
y.dep <- 13

#independent variables (dropping ID variables)
x.indep <- c(2:12)

#LOgistic regression in H20
logistic.model <- h2o.glm( y = y.dep, x = x.indep, training_frame = train.h2o, family = "binomial")
h2o.performance(logistic.model)
h2o.gainsLift(logistic.model)
h2o.varimp(logistic.model)
predict.log <- as.data.frame(h2o.predict(logistic.model, valid.h2o))
prediction = ifelse(as.character(predict.log$predict)=="N",1,0)
Actual = ifelse(valid.data$Loan_Status=="N",1,0)
confusionMatrix(predict.log$predict, valid.data$Loan_Status, positive = "N")

pred <- ROCR::prediction(prediction, Actual)
perf <- ROCR::performance(pred, "tpr", "fpr")
plot(perf,col="black",lty=3, lwd=3)
precision <- posPredValue(predict.log$predict, valid.data$Loan_Status, positive = 'N')
recall <- sensitivity(predict.log$predict, valid.data$Loan_Status, positive = 'N')
F1 <- (2 * precision * recall) / (precision + recall)
Logistic = cbind(precision, recall , F1)
mean(predict.log$predict == valid.data$Loan_Status)

 #making prediction and writing submission file
test.h2o <- as.h2o(test.data1)
predict.log2 <- as.data.frame(h2o.predict(logistic.model, test.h2o))
logi2 <- data.frame(Loan_ID = test.data1$Loan_ID, Loan_Status = predict.log2$predict)
write.csv(logi2, file = "logi2.csv", row.names = F)

#RF H20
system.time(rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, mtries = 3, max_depth = 4, seed = 1122) )
h2o.performance(rforest.model)
h2o.gainsLift(rforest.model)
h2o.varimp(rforest.model)
predict.RF <- as.data.frame(h2o.predict(rforest.model, valid.h2o))
prediction = ifelse(as.character(predict.RF$predict)=="N",1,0)
Actual = ifelse(valid.data$Loan_Status=="N",1,0)
confusionMatrix(predict.RF$predict, valid.data$Loan_Status, positive = "N")
mean(predict.RF$predict == valid.data$Loan_Status)

#making prediction and writing submission file
test.h2o <- as.h2o(test.data1)
predict.RF2 <- as.data.frame(h2o.predict(rforest.model, test.h2o))
RF2 <- data.frame(Loan_ID = test.data1$Loan_ID, Loan_Status = predict.RF2$predict)
write.csv(RF2, file = "RF2.csv", row.names = F)


#GBM H20
system.time(gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, max_depth = 4, nfolds = 4, learn_rate = 0.01, seed = 1122))
h2o.performance(gbm.model)
h2o.gainsLift(gbm.model)
h2o.varimp(gbm.model)
predict.GBM <- as.data.frame(h2o.predict(gbm.model, valid.h2o))
prediction = ifelse(as.character(predict.GBM$predict)=="N",1,0)
Actual = ifelse(valid.data$Loan_Status=="N",1,0)
confusionMatrix(predict.GBM$predict, valid.data$Loan_Status, positive = "N")
mean(predict.GBM$predict == valid.data$Loan_Status)

#making prediction and writing submission file
test.h2o <- as.h2o(test.data1)
predict.GBM2 <- as.data.frame(h2o.predict(gbm.model, test.h2o))
GBM2 <- data.frame(Loan_ID = test.data1$Loan_ID, Loan_Status = predict.GBM2$predict)
write.csv(GBM2, file = "GBM2.csv", row.names = F)
