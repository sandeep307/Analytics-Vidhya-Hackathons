rm(list=ls())
setwd ("../StudentHunt")

training =  read.csv("StudentHunt/train.csv")
test =  read.csv("StudentHunt/test.csv")
test$Footfall = 0

data =  rbind(training,test)
sapply(data, function(x){sum(is.na(x))   })
sapply(data, function(x){length(unique(x))   })

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
data$Park_ID = as.factor(data$Park_ID)
data$Location_Type= as.factor(data$Location_Type)
data$Month = as.factor(data$Month)
data$weekday = as.factor(data$weekday)
data1 = data[,-c(1,3)]

#Separating train and test datset
dev.data = data1[1:nrow(training),]
dev.data$Park_ID[dev.data$Park_ID == "19"] = NA
parkIDs = as.vector(na.omit(unique(dev.data$Park_ID)))
dev.data$Park_ID[is.na(dev.data$Park_ID)] = sample(parkIDs,sum(is.na(dev.data$Park_ID)), replace = T)
test.data = data1[(nrow(training)+1):nrow(data1),]


library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.clusterIsUp()

#data to h2o cluster
train.h2o <- as.h2o(dev.data)
test.h2o <- as.h2o(test.data)

#check column index number
colnames(train.h2o)
colnames(test.h2o)

#dependent variable (Footfall)
y.dep <- 16

#independent variables (dropping ID variables)
x.indep <- c(1:15,17:18)

################### Multiple regression in H2O #####################################

regression.model <- h2o.glm( y = y.dep, x = x.indep, training_frame = train.h2o, family = "gaussian")
h2o.performance(regression.model)
#make predictions
predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))
sub_reg <- data.frame(ID = test$ID, Footfall = predict.reg$predict)
write.csv(sub_reg, file = "sub_reg0.csv", row.names = F)

########################### Random Forest in H2O ##########################################

system.time(rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 200, mtries = 3, max_depth = 4, seed = 1122) )

sort(sapply(ls(), function(x) format(object.size(get(x)), unit = 'auto')))

h2o.performance(rforest.model)
#check variable importance
h2o.varimp(rforest.model)

#making predictions on unseen data
system.time(predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o)))

#writing submission file
sub_rf <- data.frame(ID = test$ID, Footfall = predict.rforest$predict)
write.csv(sub_rf, file = "sub_rf0.csv", row.names = F)

############################# GBM in H2O #################################################

system.time(gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, max_depth = 4, nfolds = 4, learn_rate = 0.01, sample_rate = 0.8, col_sample_rate = 0.8, seed = 1122))
h2o.performance (gbm.model)
gbm.model@model$cross_validation_metrics_summary
h2o.mse(h2o.performance(gbm.model, xval = TRUE))

#GBM Tuning
splits <- h2o.splitFrame( data = train.h2o, ratios = c(0.8), destination_frames = c("train.hex", "valid.hex"), seed = 1234 )
train <- splits[[1]]
valid <- splits[[2]]

system.time(gbm.tuned_model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train, validation_frame = valid, ntrees = 1000, learn_rate = 0.01, stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "MSE", score_tree_interval = 10, sample_rate = 0.8, col_sample_rate = 0.8, seed = 1122))
h2o.performance (gbm.tuned_model)

#making prediction and writing submission file
predict.gbm <- as.data.frame(h2o.predict(gbm.tuned_model, test.h2o))
sub_gbm <- data.frame(ID = test$ID, Footfall = predict.gbm$predict)
write.csv(sub_gbm, file = "sub_gbm3.csv", row.names = F)

################################# Deep Learning in H2O ##########################################

#DL Tuning
splits <- h2o.splitFrame( data = train.h2o, ratios = c(0.8), destination_frames = c("train.hex", "valid.hex"), seed = 1234 )
train <- splits[[1]]
valid <- splits[[2]]

model <- h2o.deeplearning(
  training_frame=train, 
  validation_frame=valid, 
  x=x.indep, 
  y=y.dep, 
  overwrite_with_best_model=F,    ## Return the final model after 10 epochs, even if not the best
  hidden=c(100,100),          ## more hidden layers -> more complex interactions
  epochs=10,                      ## to keep it short enough
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  rate=0.01, 
  rate_annealing=2e-6,            
  momentum_start=0.2,             ## manually tuned momentum
  momentum_stable=0.4, 
  momentum_ramp=1e7, 
  l1=1e-5,                        ## add some L1/L2 regularization
  l2=1e-5,
  max_w2=10                       ## helps stability for Rectifier
) 
summary(model)

h2o.performance(model, train=T)          ## sampled training data (from model building)
h2o.performance(model, valid=T)          ## sampled validation data (from model building)
h2o.performance(model, newdata=train)    ## full training data
h2o.performance(model, newdata=valid)    ## full validation data

#create a data frame and writing submission file
predict.dl2 <- as.data.frame(h2o.predict(model, test.h2o))
sub_dlearning <- data.frame(ID = test$ID, Footfall = predict.dl2$predict)
write.csv(sub_dlearning, file = "sub_dlearning3.csv", row.names = F)
