rm(list=ls())
library(dummies)
library(gmodels)
library(data.table)
library(ggplot2)

#load data using fread

train <- fread("train.csv", stringsAsFactors = T)
test <- fread("test.csv", stringsAsFactors = T)

#first prediction using mean
sub_mean = data.table(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = mean(train$Purchase))
write.csv(sub_mean, file = "first_sub.csv", row.names = F)


#combine data set
test[,Purchase :=mean(train$Purchase)]
c <- list(train,test)
combin <- rbindlist(c)

#data exploration
combin[,prop.table(table(Gender))]
combin[,prop.table(table(Age))]
combin[,prop.table(table(City_Category))]
combin[,prop.table(table(Stay_In_Current_City_Years))]
combin[,prop.table(table(Gender))]

#unique values in ID variables
length(unique(combin$Product_ID))
length(unique(combin$User_ID))
colSums(is.na(combin))

#bivariate analysis
#Age vs Gender
ggplot(combin, aes(Age,fill=Gender))+geom_bar()
ggplot(combin, aes(Age,fill=City_Category))+geom_bar()
CrossTable(combin$Occupation,combin$City_Category)

#Data Manipulation
#Create new variables for missing values
combin[,Product_Category_2_NA := ifelse(is.na(Product_Category_2)==T,1,0)]
combin[,Product_Category_3_NA := ifelse(is.na(Product_Category_3)==T,1,0)]

#impute missing values
combin[,Product_Category_2 := ifelse(is.na(Product_Category_2) == TRUE, "-999",  Product_Category_2)]
combin[,Product_Category_3 := ifelse(is.na(Product_Category_3) == TRUE, "-999",  Product_Category_3)]

#impute missing values
combin[,Product_Category_2 := ifelse(is.na(Product_Category_2) == TRUE, "-999", Product_Category_2)]
combin[,Product_Category_3 := ifelse(is.na(Product_Category_3) == TRUE, "-999",  Product_Category_3)]


#set column level
levels(combin$Stay_In_Current_City_Years)[levels(combin$Stay_In_Current_City_Years) ==  "4+"] <- "4"

#recoding age groups
levels(combin$Age)[levels(combin$Age) == "0-17"] <- 0
levels(combin$Age)[levels(combin$Age) == "18-25"] <- 1
levels(combin$Age)[levels(combin$Age) == "26-35"] <- 2
levels(combin$Age)[levels(combin$Age) == "36-45"] <- 3
levels(combin$Age)[levels(combin$Age) == "46-50"] <- 4
levels(combin$Age)[levels(combin$Age) == "51-55"] <- 5
levels(combin$Age)[levels(combin$Age) == "55+"] <- 6

#convert age to numeric
combin$Age <- as.numeric(combin$Age)

#convert Gender into numeric
combin[, Gender := as.numeric(as.factor(Gender)) - 1]

#User Count
combin[, User_Count := .N, by = User_ID]

#Product Count
combin[, Product_Count := .N, by = Product_ID]

#Mean Purchase of Product
combin[, Mean_Purchase_Product := mean(Purchase), by = Product_ID]

#Mean Purchase of User
combin[, Mean_Purchase_User := mean(Purchase), by = User_ID]

#OHE
combin <- dummy.data.frame(combin, names = c("City_Category"), sep = "_")

#check classes of all variables
sapply(combin, class)
#converting Product Category 2 & 3
combin$Product_Category_2 <- as.integer(combin$Product_Category_2)
combin$Product_Category_3 <- as.integer(combin$Product_Category_3)

#Model Building
#Divide into train and test
c.train <- combin[1:nrow(train),]
c.test <- combin[-(1:nrow(train)),]
c.train <- c.train[c.train$Product_Category_1 <= 18,]
library(h2o)
localH2O <- h2o.init(nthreads = -1)

#data to h2o cluster
train.h2o <- as.h2o(c.train)
test.h2o <- as.h2o(c.test)

#check column index number
colnames(train.h2o)
#dependent variable (Purchase)
y.dep <- 14

#independent variables (dropping ID variables)
x.indep <- c(3:13,15:20)

#Multiple regression in H20

regression.model <- h2o.glm( y = y.dep, x = x.indep, training_frame = train.h2o, family = "gaussian")
h2o.performance(regression.model)
#make predictions
predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))
sub_reg <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase =  predict.reg$predict)
write.csv(sub_reg, file = "sub_reg.csv", row.names = F)

#RF in H20
#Random Forest
system.time(rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 200, mtries = 3, max_depth = 4, seed = 1122) )

sort(sapply(ls(), function(x) format(object.size(get(x)), unit = 'auto')))

h2o.performance(rforest.model)
#check variable importance
h2o.varimp(rforest.model)

#making predictions on unseen data
system.time(predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o)))

#writing submission file
sub_rf <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase =  predict.rforest$predict)
write.csv(sub_rf, file = "sub_rf.csv", row.names = F)

#GBM in H2O
#GBM
system.time(gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1122))
h2o.performance (gbm.model)

#making prediction and writing submission file
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbm <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.gbm$predict)
write.csv(sub_gbm, file = "sub_gbm.csv", row.names = F)

#Deep Learning in H2O
#deep learning models
system.time(dlearning.model <- h2o.deeplearning(y = y.dep,x = x.indep,training_frame = train.h2o,epoch = 60,hidden = c(100,100),activation = "Rectifier",seed = 1122))
h2o.performance(dlearning.model)

#making predictions
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))

#create a data frame and writing submission file
sub_dlearning <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.dl2$predict)
write.csv(sub_dlearning, file = "sub_dlearning_new.csv", row.names = F)
