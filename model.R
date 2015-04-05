training_file = "C:\\Users\\prashkr\\Documents\\8th Sem\\MLT\\Project\\train.csv"
testing_file = "C:\\Users\\prashkr\\Documents\\8th Sem\\MLT\\Project\\test.csv"
result_file = "C:\\Users\\prashkr\\Documents\\8th Sem\\MLT\\Project\\resultFolder\\craniumCrushers_v3.csv"
result_file_gbm = "C:\\Users\\prashkr\\Documents\\8th Sem\\MLT\\Project\\resultFolder\\craniumCrushers_gbm_v1.csv"
result_file_nnet = "C:\\Users\\prashkr\\Documents\\8th Sem\\MLT\\Project\\resultFolder\\craniumCrushers_nnet_v1.csv"

library(caret)


CEVM <- function(data.trainset){
  
  mean.frame = aggregate(cbind(temp,atemp,humidity,windspeed,casual,registered,count) ~ hour,data.trainset,mean)
  mean.frame$count = as.integer(mean.frame$count)
  mean.frame$registered = as.integer(mean.frame$registered)
  mean.frame$casual = as.integer(mean.frame$casual)
  
  return (mean.frame)
}

factorization <- function(data.trainset){
  
  data.trainset$season = factor(data.trainset$season)
  data.trainset$holiday = factor(data.trainset$holiday)
  data.trainset$workingday = factor(data.trainset$workingday)
  data.trainset$weather = factor(data.trainset$weather)
  
  data.trainset$date = substring(data.trainset$datetime,1,10)
  data.trainset$hour = as.integer(substring(data.trainset$datetime,12,13))
  data.trainset$year = as.integer(substring(data.trainset$date,0,4))
  data.trainset$month = as.numeric(format(as.Date(data.trainset$date), "%m"))
  data.trainset$weekday = as.integer(as.POSIXlt(data.trainset$date)$wday)	
  #write.csv(data.trainset,"trainFeaturised.csv",row.names=FALSE)
  data.trainset$month = factor(data.trainset$month)
  data.trainset$weekday = factor(data.trainset$weekday)	
  data.trainset$hour = factor(data.trainset$hour)
  data.trainset$year = factor(data.trainset$year)
  
  return (data.trainset)
  
}

rf <- function(maxTree,data.trainset,data.testset){
  
  library(randomForest)
  data.trainset$casual = as.integer(data.trainset$casual)
  data.trainset$registered = as.integer(data.trainset$registered)
  data.trainset$count = as.integer(data.trainset$count)
  
  
  fit_casual <- randomForest(casual ~  season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,data=data.trainset,ntree=maxTree)
  fit_reg <- randomForest(registered ~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,data=data.trainset,ntree=maxTree)
  #fit_count <- randomForest(count ~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed+casual+registered+hour+year+month+weekday,data=data.trainset,ntree=maxTree)
  
  
  casual.pred <- predict(fit_casual, data.testset)
  reg.pred <- predict(fit_reg, data.testset)
  #print (casual.pred)
  count.pred = round(casual.pred + reg.pred)
  
  print (head(count.pred))
  submission = data.frame(datetime = data.testset$datetime, count=count.pred)
  write.csv(submission, file = result_file,row.names=FALSE)
  #fit_count
  #print(paste("OOB Error is ",fit_count$err.rate[maxTree]))
}

ctrees <- function(data.trainset,data.testset){
  library(party)
  data.trainset$casual = as.integer(data.trainset$casual)
  data.trainset$registered = as.integer(data.trainset$registered)
  data.trainset$count = as.integer(data.trainset$count)
  
  fit_count <- ctree(count ~  season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,data=data.trainset)
  count.pred <- predict(fit_count, data.testset)
  #reg.pred <- predict(fit_reg, data.testset)
  count.pred = as.integer(count.pred)
  print (head(count.pred))
  submission = data.frame(datetime = data.testset$datetime, count=count.pred)
  write.csv(submission, result_file,row.names=FALSE)
  
}


gradientBoosting <- function(data.trainset,data.testset){
  library(gbm)
  
  data.trainset$casual = as.integer(data.trainset$casual)
  data.trainset$registered = as.integer(data.trainset$registered)
  data.trainset$count = as.integer(data.trainset$count)
  
#   gbm.model <- gbm(count ~  season+holiday+workingday+weather+atemp+humidity+windspeed+hour+year+month+weekday,
#                  data=data.trainset,
#                  var.monotone=NULL,
#                  distribution="gaussian",
#                  n.trees = 1000,
#                  shrinkage=0.05,
#                  n.minobsinnode = 10,
#                  interaction.depth = 3,
#                  bag.fraction = 0.5,
#                  train.fraction = 1,
#                  verbose=TRUE
#   )
#   count.pred <- predict(gbm.fit, data.testset, 500)
##########USING CARET LIBRARY###########
  gbm.control <- trainControl(method='cv', number=10, returnResamp='none')  
  
  gbm.model <- train(count ~  season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,
                    data=data.trainset, 
                    method='gbm', 
                    trControl=gbm.control
                    )
  

#   best.iter <- gbm.perf(gbm.model,method="cv")
  count.pred <- predict(object=gbm.model, data.testset, best.iter, type="response")
  print (head(count.pred))
  count.pred[count.pred<0] <- 0
# 
#   
  count.pred = as.integer(count.pred)
  submission = data.frame(datetime = data.testset$datetime, count=count.pred)
  write.csv(submission, file=result_file_gbm,row.names=FALSE)
}

neuralNet <- function(data.trainset,data.testset){
  library(nnet) # or can use caret
  library(neuralnet)
#   data.trainset$casual = as.integer(data.trainset$casual)
#   data.trainset$registered = as.integer(data.trainset$registered)
#   data.trainset$count = as.integer(data.trainset$count)
  #using caret lib
#   model <- train(count ~  season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,
#                  data=data.trainset,
#                  method='nnet',
#                  linout=TRUE,
#                  trace = FALSE
#                  #Grid of tuning parameters to try:
#                  #tuneGrid=expand.grid(.size=c(1,5,10),.decay=c(0,0.001,0.1))
#                  )
#   
#   count.pred <- predict(model, data.testset)
#   print (head(count.pred))
#   count.pred[count.pred<0] <- 0
  #using nnet library

   formula = count ~ season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday

#   model <- nnet(count ~  season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,
#                 data = data.trainset,
#                 size = 12,
#                 decay = 1e-3
#                 )
#   # 
#   # 
  
#   formula <- count ~ season2+season3+season4+workingday1+weather2+weather3+year2012
#                      +hour1+hour2+hour3+hour4+hour5+hour6+hour7+hour8+hour9+hour10
#                      +hour11+hour12+hour13+hour14+hour15+hour16+hour17+hour18+hour19
#                      +hour20+hour21+hour22+hour23+weekday2+weekday3+weekday4+weekday5
#                      +weekday6+weekday7

  model <- neuralnet(formula
                   ,data=data.trainset
                   ,hidden=c(7,8,9,8,7)
                   ,threshold=.04
                   ,stepmax=1e+06
                   ,learningrate=.001
                   ,algorithm="rprop+"
                   ,lifesign="full"
                   ,likelihood=T
                   )
  
  
  count.pred <- predict(model, data.testset)
  print (head(count.pred))
  count.pred = as.integer(count.pred)
  submission = data.frame(datetime = datetimeCol, count=count.pred)
  write.csv(submission, file=result_file_nnet,row.names=FALSE)
  
}


data.trainset <- read.csv(training_file, header=TRUE)
names(data.trainset) <- c("datetime","season","holiday","workingday","weather","temp","atemp","humidity","windspeed","casual","registered","count")
data.trainset = factorization(data.trainset)

data.testset <- read.csv(testing_file, header=TRUE)
names(data.testset) <- c("datetime","season","holiday","workingday","weather","temp","atemp","humidity","windspeed")
data.testset = factorization(data.testset)

datetimeCol = data.testset$datetime

data.trainset$datetime <- NULL
data.testset$datetime <- NULL

# countCol <- data.trainset$count
# 
# trainmat <- model.matrix(count~season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,data=data.trainset)
# 
# testmat <- model.matrix(~season+holiday+workingday+weather+temp+atemp+humidity+windspeed+hour+year+month+weekday,data=data.testset)
# 
# data.trainset <- as.data.frame(trainmat)
# 
# data.testset <- as.data.frame(testmat)
# 
# data.trainset$count <- cbind(data.trainset, countCol)


#rf(100,data.trainset,data.testset)
#ctrees(data.trainset,data.testset)
#gradientBoosting(data.trainset,data.testset)
neuralNet(data.trainset, data.testset)
