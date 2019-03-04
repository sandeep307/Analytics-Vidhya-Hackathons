#Loading libraries
rm(list=ls())
library(dplyr)
library(caret)
library(xgboost)

#Reading the data
setwd("../Knocktober")

#Profile data
profile = read.csv("Knocktober/Patient_Profile.csv", stringsAsFactors = F, na.strings = c("", "NA", NA))
profile = profile[,-c(6:8,10:11)]

#Combining HCs results
HC_details = read.csv("Knocktober/Health_Camp_Detail.csv", stringsAsFactors = F, na.strings = c("", "NA", NA))
First_HC = read.csv("Knocktober/First_Health_Camp_Attended.csv", stringsAsFactors = F, na.strings = c("", "NA", NA))
Second_HC = read.csv("Knocktober/Second_Health_Camp_Attended.csv", stringsAsFactors = F, na.strings = c("", "NA", NA))
Third_HC = read.csv("Knocktober/Third_Health_Camp_Attended.csv", stringsAsFactors = F, na.strings = c("", "NA", NA))
Second_HC$Donation = NA
Second_HC =  cbind(Second_HC[, c(1:2)], Second_HC[, 4], Second_HC[, 3])
names(Second_HC) = names(First_HC)

F_S_HC = rbind(First_HC, Second_HC)
F_S_HC$Number_of_stall_visited = NA
Third_HC$Donation = NA
Third_HC$Health_Score = NA
Third_HC$Last_Stall_Visited_Number = NULL

Third_HC = cbind(Third_HC[,-3], Third_HC[,3] )
names(Third_HC) = names(F_S_HC)
HC = rbind(F_S_HC, Third_HC)

#Combining Train and test data
train_data = read.csv("Knocktober/train.csv", stringsAsFactors = F, na.strings = c("", "NA", NA))
test_data = read.csv("Knocktober/test.csv", stringsAsFactors = F, na.strings = c("", "NA", NA))
data = rbind(train_data, test_data)

#combining patients and health camp data
data1 = dplyr::left_join(data, HC, by = c("Patient_ID","Health_Camp_ID") )
data2 = dplyr::left_join(data1, profile, by = "Patient_ID")
data3 = dplyr::left_join(data2, HC_details, by = "Health_Camp_ID")


#Feature Engineering
data3$Registration_Date = as.Date(data3$Registration_Date, "%d-%B-%y")
data3$Camp_Start_Date = as.Date(data3$Camp_Start_Date, "%d-%B-%y")
data3$Camp_End_Date = as.Date(data3$Camp_End_Date, "%d-%B-%y")
data3$First_Interaction = as.Date(data3$First_Interaction, "%d-%B-%y")

#Days left for the camp to end after registration
data3$DaysLeft = as.numeric(data3$Camp_End_Date - data3$Registration_Date)

#registration month and day
data3$Registration_Month = months.Date(data3$Registration_Date)
data3$Registration_Day = weekdays(data3$Registration_Date)
data3$Isweekend = ifelse(data3$Registration_Day %in% c("Saturday", "Sunday"), 1, 0)

#days Since First Interaction and Camp Duration
data3$DSFI = as.numeric(data3$Registration_Date - data3$First_Interaction)
data3$Camp_Duration = as.numeric(data3$Camp_End_Date - data3$Camp_Start_Date)

HC_patients = dplyr::select(data3, Patient_ID, Health_Camp_ID) %>% group_by(Health_Camp_ID) %>% summarise(Camp_patients = n() )
data3 = dplyr::left_join(data3, HC_patients, by = "Health_Camp_ID")
data3 = dplyr::mutate(data3, Total_Shared = LinkedIn_Shared + Twitter_Shared + Facebook_Shared)

#Creating response variable
data3$showup = as.numeric( !is.na(data3$Health_Score) | (!is.na(data3$Number_of_stall_visited) & data3$Number_of_stall_visited > 0 ) )

#preparing final data set
data4 = data3[,-c(1:3, 9:11, 16:18)]

#Changing chracter to factor variable
for(i in names(data4))
{
  if(class(data4[[i]]) == "character")
    data4[[i]] = as.factor(data4[[i]])
}

#One Hot Encoding
ohe_feats <- c("Category1" , "Category2" , "Registration_Month" , "Registration_Day")
dummies <- dummyVars(~ Category1 + Category2 + Registration_Month + Registration_Day, data = data4) 
data4_ohe <- as.data.frame(predict(dummies, newdata = data4))
data4_combined <- cbind( data4[,-c(which(colnames(data4) %in% ohe_feats))],data4_ohe)

#Separating train and test data 
test_PID = test_data$Patient_ID
test_HCID = test_data$Health_Camp_ID

#Tuning XGB Model

X_train = data4_combined[1:75278, -17]
Y_train = data4_combined[1:75278, 17]
X_test = data4_combined[75279:nrow(data4),-17]

best_auc = 0
best_auc_index = 0

for (iter in 1:20) {
  print(iter)
  param <- list(objective = "binary:logistic",
                eval_metric = "auc",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=as.matrix(X_train), label=as.matrix(Y_train), params = param, nthread=6,  nfold = cv.nfold, missing = "NAN", nrounds=cv.nround, verbose = T, early.stop.round = 20, maximize = T)
  
  max_auc = max(mdcv$evaluation_log$test_auc_mean)
  max_auc_index = which.max(mdcv$evaluation_log$test_auc_mean)
  
  if (max_auc > best_auc) {
    best_auc = max_auc
    best_auc_index = max_auc_index
    best_seednumber = seed.number
    best_param = param
  }
}

nround = best_auc_index
set.seed(best_seednumber)
model_xgb1 <- xgb.train(data = xgb.DMatrix(as.matrix(X_train), label = as.matrix(Y_train), missing = "NAN"), params=best_param, missing = "NAN" , nrounds=nround, verbose = T, nthread=6)

#prediction
pred_train <- predict(model_xgb1, as.matrix( X_test ),missing = "NAN" )
df = data.frame(Patient_ID =  test_PID, Health_Camp_ID  = test_HCID, Outcome= pred_train)
write.csv(df, file = "xgb8.csv", row.names = F)

#variable importance
xgb.importance(names(data4_combined[,-17]),model = model_xgb1)
# xgb.plot.tree(names(data4_combined[,-17]),model = model_xgb1)
# xgb.plot.importance(xgb.importance(names(data4_combined[,-17]),model = model_xgb1))


