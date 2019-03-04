rm(list = ls() )
require(glmnet)
require(caret)
require(ggplot2)
require(dplyr)
setwd ("../StudentHunt")

training =  read.csv("StudentHunt/train.csv")
test =  read.csv("StudentHunt/test.csv")
test$Footfall = 0
data =  rbind(training,test)

sapply(data, function(x){sum(is.na(x))   })
sapply(data, function(x){length(unique(x))   })

# Imputing Missing Values
data$Direction_Of_Wind[is.na(data$Direction_Of_Wind)] = mean(data$Direction_Of_Wind, na.rm = T)
data$Average_Breeze_Speed[is.na(data$Average_Breeze_Speed)] = mean(data$Average_Breeze_Speed, na.rm = T)
data$Max_Breeze_Speed[is.na(data$Max_Breeze_Speed)] = mean(data$Max_Breeze_Speed, na.rm = T)
data$Min_Breeze_Speed[is.na(data$Min_Breeze_Speed)] = mean(data$Min_Breeze_Speed, na.rm = T)
data$Average_Atmospheric_Pressure[is.na(data$Average_Atmospheric_Pressure)] = mean(data$Average_Atmospheric_Pressure, na.rm = T)
data$Max_Atmospheric_Pressure[is.na(data$Max_Atmospheric_Pressure)] = mean(data$Max_Atmospheric_Pressure, na.rm = T)
data$Min_Atmospheric_Pressure[is.na(data$Min_Atmospheric_Pressure)] = mean(data$Min_Atmospheric_Pressure, na.rm = T)
data$Max_Ambient_Pollution[is.na(data$Max_Ambient_Pollution)] = mean(data$Max_Ambient_Pollution, na.rm = T)
data$Min_Ambient_Pollution[is.na(data$Min_Ambient_Pollution)] = mean(data$Min_Ambient_Pollution, na.rm = T)
data$Average_Moisture_In_Park[is.na(data$Average_Moisture_In_Park)] = mean(data$Average_Moisture_In_Park, na.rm = T)
data$Max_Moisture_In_Park[is.na(data$Max_Moisture_In_Park)] = mean(data$Max_Moisture_In_Park, na.rm = T)
data$Min_Moisture_In_Park[is.na(data$Min_Moisture_In_Park)] = mean(data$Min_Moisture_In_Park, na.rm = T)
data$Var1[is.na(data$Var1)] = mean(data$Var1, na.rm = T)
data$Location_Type  = as.character(data$Location_Type)
data$Park_ID  = as.character(data$Park_ID)
data$Date = as.character(data$Date)
data$Date = as.Date(data$Date, "%d-%m-%Y")
data$Month = (months.Date(data$Date))
data$weekday = as.character(weekdays(data$Date))

data1 = data[,-c(1,3)]

#Separating train and test datset
dev.data = data1[1:nrow(training),]
dev.data$Park_ID[dev.data$Park_ID == "19"] = NA
parkIDs = as.vector(na.omit(unique(dev.data$Park_ID)))
dev.data$Park_ID[is.na(dev.data$Park_ID)] = sample(parkIDs,sum(is.na(dev.data$Park_ID)), replace = T)

test.data = data1[(nrow(training)+1):nrow(data1),]
depVar = paste(names(select(dev.data,-Footfall)), collapse = "+")
formula <- as.formula(paste("Footfall ~", depVar))

#Lasso and Ridge Regression
x = model.matrix(formula, dev.data)[,-1]
y = dev.data$Footfall
set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)

#Ridge Regression
library(glmnet)
ridge.mod = glmnet(x,y,alpha = 0)
dim(coef(ridge.mod))
plot(ridge.mod)
set.seed (1)
cv.out.ridge = cv.glmnet (x[train ,],y[train],alpha =0)
plot(cv.out.ridge)
bestlam = cv.out.ridge$lambda.min
options(digits=6)
ridge.pred = predict (ridge.mod ,s = bestlam ,newx = x[-train,])
sqrt(mean(( ridge.pred -y[-train])^2))

#Refit on full dataset using min lambda
out.ridge=glmnet (x,y,alpha =0)
plot(out.ridge)
ridge.coef = predict (out.ridge ,type="coefficients",s = bestlam )
ridge.coef

#prediction final using Ridge
x_test = model.matrix(formula, test.data)[,-1]
ridge.pred_test = predict (out.ridge ,s = bestlam ,newx = x_test)
df_ridge = data.frame(ID = test$ID, Footfall = ridge.pred_test)
names(df_ridge)= c("ID", "Footfall")
write.csv(df_ridge, file = "sol_ridge.csv", row.names = F)

#THE LASSO
depVar = paste(names(select(dev.data,-Footfall)), collapse = "+")
formula <- as.formula(paste("Footfall ~", depVar))
x = model.matrix(formula, dev.data)[,-1]
y = dev.data$Footfall
set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
grid =10^ seq (10,-2, length =100)
lasso.mod = glmnet (x[train,], y[train], alpha =1, lambda = grid)

plot(lasso.mod)
par(mfrow=c(1,2))
set.seed (1)
cv.out.lasso =cv.glmnet (x[train ,],y[train],alpha =1)
plot(cv.out.lasso)
bestlam = cv.out.lasso$lambda.min
lasso.pred = predict (lasso.mod ,s = bestlam ,newx = x[-train ,])
sqrt(mean(( lasso.pred -y[-train])^2))

#Refit Lasso for Entire Dataset
out.lasso = glmnet (x, y, alpha =1)
plot(out.lasso)
lasso.coef = predict (out.lasso ,type ="coefficients",s = bestlam )
lasso.coef
summary(lasso.mod)


#prediction final
x_test = model.matrix(formula, test.data)[,-1]
lasso.pred_test = predict (out.lasso ,s = bestlam ,newx = x_test)
df_lasso = data.frame(ID = test$ID, Footfall = lasso.pred_test)
names(df_lasso)= c("ID", "Footfall")
write.csv(df_lasso, file = "sol_lasso.csv", row.names = F)