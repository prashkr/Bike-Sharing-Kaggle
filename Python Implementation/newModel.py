# -*- coding: utf-8 -*-
import math
import sys
import numpy as np
import datetime
from sklearn import ensemble
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
import scipy as sp

count_season = [0]*4
count_season[0] = 116
count_season[1] = 215
count_season[2] = 234
count_season[3] = 198

count_holiday = [0]*2

count_holiday[0] = 191
count_holiday[1] = 185

cols_to_use = (0,4,8,9,10,11,12,13,14,15,16)

def featureEngineeringHelper(row, isTrainData):
	global count_season
	global count_holiday
	row = row.split(',')
	(date, time) = row[0].split(' ')
	year = int(date.split('-')[0])
	month = int(date.split('-')[1])
	day = int(date.split('-')[2])
	weekday = datetime.datetime(year, month, day).weekday()
	year = year - 2011
	hour = int(time.split(':')[0])
	#hour = math.cos(((2*math.pi)/24)*hour)
	season = int(row[1])
	spring =0 
	summer=0
	fall=0

	if(season == 1):
		spring=1
	elif(season == 2):
		summer = 1
	elif(season == 3):
		fall = 1
	else:
		pass

	holiday = int(row[2])
	workingDay = int(row[3])
	weather = int(row[4])
	if(weather == 4):
		weather = 3
	temp = float(row[5])
	atemp = float(row[6])
	humidity = float(row[7])
	windspeed = float(row[8])

	if(workingDay == 1 and ((hour >= 7 and hour <= 9) or (hour >= 17 or hour <= 19))):
		peakhour = 1
	else:
		peakhour = 0

	if(workingDay == 0 and ((hour >= 10 and hour <= 18))):
		peakhour = 1
	else:
		peakhour = 0
	if isTrainData:
		casual = int(row[9])
		registered = int(row[10])
		count = int(row[11])	
		#count_season[season-1] = count_season[season-1]+ count
		#season_c[season-1] = season_c[season-1]+1
		#count_holiday[holiday] += count
		#holiday_c[holiday] += 1

		#                    0     1      2     3       4      5       6      7      8       	9        10       11     12      13        14       15		        16                         17
		transformed_row = [year, month, day, weekday, hour, spring, summer, fall ,holiday, workingDay, weather, temp, atemp, humidity, windspeed, peakhour, count_season[season-1], count_holiday[holiday], casual, registered]
		# transformed_row = [year, month, day, weekday, hour, count_season[season-1] ,holiday, workingDay, weather, temp, atemp, humidity, windspeed, peakhour, casual, registered]
		#month used here

		return transformed_row
	else:
		# transformed_row = [year, month, day, weekday, hour, spring, summer, fall, holiday, workingDay, weather, temp, atemp, humidity, windspeed, peakhour]	
		transformed_row = [year, month, day, weekday, hour, spring, summer, fall ,holiday, workingDay, weather, temp, atemp, humidity, windspeed, peakhour, count_season[season-1], count_holiday[holiday]]
		return transformed_row

def featureEngineering(dataSet, isTrainData):
	dataSet.readline() #Skipping the headers
	dataSetModified = []
	while True:
		row = dataSet.readline()
		if not row:
			break
		transformed_row = featureEngineeringHelper(row, isTrainData) #y1 is a tuple of the form (casual, registered) 												 #x1 is a list of new features
		dataSetModified.append(transformed_row)
	return dataSetModified     

def getRMLSE(Y_predicted):
	Y_actual = np.load("answers.npy")
	return score_func(Y_actual, Y_predicted)

def score_func(y, y_pred):
	y = y.ravel()
	y_pred = y_pred.ravel()
	res = math.sqrt( np.sum( np.square(np.log(y_pred+1) - np.log(y+1)) ) / len(y) )
	return res

def llfun(act, pred):
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
	ll = ll * -1.0/len(act)
	return ll

def cross_validation(train, target):
	#read in  data, parse into training and target sets
	#dataset = np.genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]    

	#In this case we'll use a random forest, but this could be any classifier
	params = {'n_estimators': 1000, 'random_state': 0, 'min_samples_split': 11, 'oob_score': False, 'n_jobs':1 }
	clf = ensemble.RandomForestRegressor(**params)

	params = {'n_estimators': 100, 'max_depth': 6, 'random_state':0}
	gbm = ensemble.GradientBoostingRegressor(**params)

	#Simple K-Fold cross validation. 5 folds.
	cv = cross_validation.KFold(len(train), k=5, indices=False)

	#iterate through the training and test cross validation segments and
	#run the classifier on each one, aggregating the results into a list
	results = []
	for traincv, testcv in cv:
		rf1_Y_CV = cfr.fit(train[traincv], target[traincv]).predict(train[testcv])
		gbm_Y_CV = cfr.fit(train[traincv], target[traincv]).predict(train[testcv])
		results.append(llfun(target[testcv], [x[1] for x in probas]) )

	#print out the mean of the cross-validated results
	print "Results: " + str( np.array(results).mean() )

def randomForestModel():
	params = {'n_estimators': 1000, 'random_state': 0, 'min_samples_split': 11, 'oob_score': False, 'n_jobs':1 }
	rf = ensemble.RandomForestRegressor(**params)
	return rf

def gradientDescentModel():
	params = {'n_estimators': 100, 'max_depth': 6, 'random_state':0}
	gbm = ensemble.GradientBoostingRegressor(**params)
	return gbm

def extraTreesRegressor(X, Y_casual, Y_registered, testSet_final):
	params = {'n_estimators': 1000, 'random_state': 0, 'min_samples_split': 11, 'oob_score': False, 'n_jobs':1 }
	eTreeReg1 = ensemble.ExtraTreesRegressor()
	eTreeReg2 = ensemble.ExtraTreesRegressor()

	eTreeReg1.fit(X, Y_casual)
	eTreeReg2.fit(X, Y_registered)

	eTreeReg1_Y = np.exp(eTreeReg1.predict(testSet_final))-1
	eTreeReg2_Y = np.exp(eTreeReg2.predict(testSet_final))-1

	final_prediction = np.intp(np.around(eTreeReg1_Y + eTreeReg2_Y))
	return final_prediction

def elasticnet(X,Y_casual, Y_registered, testSet_final):
	alpha=0.001
	l1_ratio=0.1
	glmnet1 = linear_model.ElasticNetCV()
	glmnet2 = linear_model.ElasticNetCV()

	glmnet1.fit(X, Y_casual)
	glmnet2.fit(X, Y_registered)

	glmnet1_Y = np.exp(glmnet1.predict(testSet_final))-1
	glmnet2_Y = np.exp(glmnet2.predict(testSet_final))-1
	final_prediction = np.intp(np.around(glmnet1_Y + glmnet2_Y))
	return final_prediction

def lasso(X, Y_casual, Y_registered, testSet_final):
	alpha = 0.5
	lasso1 =linear_model.Lasso(alpha=alpha)
	lasso2 =linear_model.Lasso(alpha=alpha)

	lasso1.fit(X, Y_casual)
	lasso2.fit(X, Y_registered)

	lasso1_Y = np.exp(lasso1.predict(testSet_final))-1
	lasso2_Y = np.exp(lasso2.predict(testSet_final))-1
	final_prediction = np.intp(np.around(lasso1_Y + lasso2_Y))
	return final_prediction

# def stackingPredictor(rf,gbm):
# 	stacking = []
# 	stacking.append(rf)
# 	stacking.append(gbm)
# 	stacking = np.array(stacking)
# 	stacking = np.transpose(stacking)
# 	return stacking

# def stacking():
# 	print "--------Ensemble Stacking ---------------"
# 	print 

# 	rf1_Y_train =  rf1.predict(X)
# 	rf2_Y_train = rf2.predict(X)
# 	gbm1_Y_train = gbm1.predict(X)
# 	gbm2_Y_train = gbm2.predict(X)
	
# 	rf1_Y = rf1.predict(testSet_final)	
# 	rf2_Y = rf2.predict(testSet_final)
# 	gbm1_Y = gbm1.predict(testSet_final)
# 	gbm2_Y = gbm2.predict(testSet_final)

# 	stacking_cas_train = stackingPredictor(rf1_Y_train , gbm1_Y_train )
# 	stacking_cas_test = stackingPredictor(rf1_Y , gbm1_Y )
	
# 	stacking_reg_train = stackingPredictor(rf2_Y_train , gbm2_Y_train )
# 	stacking_reg_test = stackingPredictor(rf2_Y, gbm2_Y)
	
# 	#clf = svm.SVR()

# 	stacker_cas= linear_model.LinearRegression()
# 	stacker_reg= linear_model.LinearRegression()
	
# 	stacker_cas.fit(stacking_cas_train,Y_casual)	
# 	stacker_reg.fit(stacking_reg_train,Y_registered)	

# 	final_prediction_cas = stacker_cas.predict(stacking_cas_test)
# 	final_prediction_reg = stacker_reg.predict(stacking_reg_test)
# 	final_prediction = (np.exp(final_prediction_cas)+1) + (np.exp(final_prediction_reg)+1)
# 	final_prediction = np.intp(np.around(final_prediction))

# 	return final_prediction

def supportVectorRegression(X, Y_casual, Y_registered, testSet_final):
	svr1 = SVR(kernel='rbf', gamma=0.1)
	svr2 = SVR(kernel='rbf', gamma=0.1)
	svr1.fit(X, Y_casual)
	svr2.fit(X, Y_registered)
	svr1_Y = np.exp(svr1.predict(testSet_final))-1
	svr2_Y = np.exp(svr2.predict(testSet_final))-1
	final_prediction = np.intp(np.around(svr1_Y + svr2_Y))
	return final_prediction

def rfGbmCombined(X, Y_casual, Y_registered, testSet_final):
	#creating models
	rf1 = randomForestModel()  #train for casual
	rf2 = randomForestModel()  #train for registered
	gbm1 = gradientDescentModel()   #train for casual
	gbm2 = gradientDescentModel()   #train for registered
	#fitting models
	# rf1.fit(train_X, train_Y[:, 0])  #train_Y[:, 0] - use 0th column of train_Y
	rf1.fit(X, Y_casual)
	rf2.fit(X, Y_registered)  
	gbm1.fit(X, Y_casual)
	gbm2.fit(X, Y_registered)

	#prediction
	rf1_Y = np.exp(rf1.predict(testSet_final))-1
	rf2_Y = np.exp(rf2.predict(testSet_final))-1
	gbm1_Y = np.exp(gbm1.predict(testSet_final))-1
	gbm2_Y = np.exp(gbm2.predict(testSet_final))-1

	#Average the prediction from classifiers
	final_prediction = (rf1_Y + rf2_Y + gbm1_Y + gbm2_Y)/2
	final_prediction = np.intp(np.around(final_prediction))  #round and convert to integer
	return final_prediction

def write_result_to_file(final_prediction, dateTimeColumn):
	f = open("submit.csv", "w")
	f.write("datetime,count\n")
	numRows = final_prediction.size
	for i in xrange(0,numRows):
		string_to_write = dateTimeColumn[i] + "," + str(final_prediction[i]) + "\n"
		f.write(string_to_write)
	f.close()

def getDateTimeColumn(testSetOriginal):
	testSetOriginal.seek(0)
	dateTimeColumn = []
	testSetOriginal.readline()
	while(True):
		row = testSetOriginal.readline()
		if not row:
			break
		row = row.split(',')
		dateTimeColumn.append(row[0])
	return dateTimeColumn

if __name__ == '__main__':
	trainSetOriginal = open('train.csv', "r")
	testSetOriginal = open('test.csv', "r")

	#Feature Engineering
	trainSetModified = featureEngineering(trainSetOriginal, True)  #numpy array format
	testSetModified = featureEngineering(testSetOriginal, False)  #numpy array format

	#splitting trainset into X and Y components. Y consists of [casual, registered]
	trainSetMod_X = []
	trainSetMod_Y = []
	for row in trainSetModified:
		trainSetMod_X.append(row[:-2])
		trainSetMod_Y.append(row[-2:])	#[casual, registered]

	#converting sets into format acceptable by the learning models
	train_X = np.array(trainSetMod_X) 
	train_Y = np.array(trainSetMod_Y) 
	testSet = np.array(testSetModified)

	Y_casual = np.log(train_Y[:, 0]+1)
	Y_registered = np.log(train_Y[:, 1]+1)

	X = train_X[:, cols_to_use]              #final train set
	testSet_final = testSet[:, cols_to_use]  #final test set

	#RandomForest and GradientBoosting Combined
	final_prediction = rfGbmCombined(X, Y_casual, Y_registered, testSet_final)

	#Elastic net model
	# final_prediction = elasticnet(X, Y_casual, Y_registered, testSet_final)

	#lasso regression model
	# final_prediction  = lasso(X, Y_casual, Y_registered, testSet_final)

	#Support Vector Regression
	# final_prediction = supportVectorRegression(X, Y_casual, Y_registered, testSet_final)

	#Extra Trees Regressor
	# final_prediction = extraTreesRegressor(X, Y_casual, Y_registered, testSet_final)

	error = getRMLSE(final_prediction)
	print error
	#get datetime column
	dateTimeColumn = getDateTimeColumn(testSetOriginal)

	#writing final_prediction to file
	write_result_to_file(final_prediction, dateTimeColumn)
