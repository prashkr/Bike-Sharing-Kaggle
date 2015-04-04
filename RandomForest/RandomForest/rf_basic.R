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
	write.csv(submission, file="craniumCrushers_v3.csv",row.names=FALSE)
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
	write.csv(submission, file="craniumCrushers_ctree_v1.csv",row.names=FALSE)

}
gradientBoosting <- function(data.trainset,data.testset){
		
}


data.trainset <- read.csv("train.csv", header=TRUE)
names(data.trainset) <- c("datetime","season","holiday","workingday","weather","temp","atemp","humidity","windspeed","casual","registered","count")
data.trainset = factorization(data.trainset)

data.testset <- read.csv("test.csv", header=TRUE)
names(data.testset) <- c("datetime","season","holiday","workingday","weather","temp","atemp","humidity","windspeed")
data.testset = factorization(data.testset)

#rf(100,data.trainset,data.testset)
ctrees(data.trainset,data.testset)