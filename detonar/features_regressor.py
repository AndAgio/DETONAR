#Python modules
import pandas as pd
import numpy as np
import math
import os
import glob
import random
from random import randint
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from statsmodels.tsa.stattools import adfuller
#from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#Python files
import settings_parser

#Get list of csv files and select those for train and those for test
def get_files_train_test(filenames, args):
	all_files = []
	for filename in filenames:
		files = glob.glob(os.path.join(filename, '*.csv'))
		all_files.append(files)
	all_files_list = [item for sublist in all_files for item in sublist]
	random.shuffle(all_files_list)
	#Split between train and test files
	train_files = all_files_list[:(int)(args.train_ratio*len(all_files_list))]
	test_files = all_files_list[(int)(args.train_ratio*len(all_files_list)):]
	print('Number of files: {} of which train: {} and test: {}'.format(len(all_files_list), len(train_files), len(test_files)))
	return train_files, test_files

def get_x_y(files, step, args):
	x = []
	y = []
	#Split training files with overlapping series like for real-time predictions
	for file in files:
		data = pd.read_csv(file)
		for index in range(0, len(data.index) - (args.lag_val + args.prediction_window) + 1, step):
			if(index+args.lag_val + args.prediction_window < len(data.index)):
				x.append(np.asarray(data.iloc[index:index+args.lag_val]))
				y.append(np.asarray(data.iloc[index+args.lag_val:index+args.lag_val+args.prediction_window]))
	x = np.asarray(x)
	y = np.asarray(y)
	#x and y: n_series x n_elements_per_series x n_features
	#x: n_series x val_lag x n_features
	#y: n_series x prediction_window x n_features
	#print('x shape: {}\ty shape: {}'.format(x.shape, y.shape))
	#print(np.asarray(x_test)[:,:,0])
	#print(np.asarray(y_test)[:,:,0])
	return x, y

#Get selected feature from the dataset
def select_features_x_y(x, y, args):
	x_feats = []
	y_feats = []
	#Produce list where each element is x and y for a single feature
	for feature in args.features_to_fit:
		index = args.all_features.index(feature)
		x_feats.append(x[:,:,index])
		y_feats.append(y[:,:,index])
	#x_feat: list of arrays of shape: n_series x val_lag
	#y_feat: list of arrays of shape: n_series x prediction_window
	#print(x_feats[0].shape)
	#print(y_feats[0].shape)	
	#print(x_feats[0])
	#print(y_feats[0])	
	return x_feats, y_feats

def standardize_windows(x_list, y_list, args):
	new_x_list = x_list
	new_y_list = y_list
	for index in range(len(x_list)):
		x=x_list[index]
		y=y_list[index]
		new_x=new_x_list[index]
		new_y=new_y_list[index]
		#print('x.shape: {} y.shape: {}'.format(x.shape, y.shape))
		for i in range(x.shape[0]):
			#seq_x = np.append(x[i,:], y[i,:])
			seq_x = x[i,:]
			#print(seq_x)
			mean = np.mean(seq_x)
			std = np.std(seq_x)
			#print('Mean={} and std={}'.format(mean,std))
			if(std==0):
				std = 1
			new_x[i,:] = (x[i,:]-mean)/std
			new_y[i,:] = (y[i,:]-mean)/std
			#print('standardized: {}'.format(new_x[i,:]))
			#if(std==1 and mean==0 and y[i,:]!=0):
			#	new_y[i,:] = y[i,:]
			#else:
			#	new_y[i,:] = (y[i,:]-mean)/std
			#print('x={} and y={}'.format(x[i,:], y[i,:]))
	return new_x_list, new_y_list

def normalize_windows(x_list, y_list, args):
	new_x_list = x_list
	new_y_list = y_list
	for index in range(len(x_list)):
		x=x_list[index]
		y=y_list[index]
		new_x=new_x_list[index]
		new_y=new_y_list[index]
		#print('x.shape: {} y.shape: {}'.format(x.shape, y.shape))
		for i in range(x.shape[0]):
			seq_x = x[i,:]
			max = np.max(seq_x)
			#print('Mean={} and std={}'.format(mean,std))
			if(max==0):
				max = 1
			new_x[i,:] = (x[i,:])/max
			new_y[i,:] = (y[i,:])/max
	return new_x_list, new_y_list

def define_model(args):
	if(args.model=='svr'):
		model = SVR(kernel='poly', degree=3)
	elif(args.model=='decision tree'):
		model = DecisionTreeRegressor(max_depth=args.max_depth)
	elif(args.model=='linear'):
		model = LinearRegression()
	elif(args.model=='random forest'):
		model = RandomForestRegressor()#bootstrap = True,max_depth = 30,max_features = 'auto',min_samples_leaf = 1,min_samples_split = 5,n_estimators = 800)
	else:
		print('Wrong model selected!!!')
		model = []
	return model

def fit_model(model, x_train, y_train, args):
	if(args.model=='svr'):
		#print(model.get_params().keys())
		parameters = {'kernel':['linear', 'rbf', 'poly'],
                      'C':[1, 5, 10],
                      'degree':[3,5,7]}
		clf = GridSearchCV(model, parameters, cv=5)
		clf.fit(x_train, y_train)
		print('Best Regressor:', clf.best_estimator_.get_params())
		fitted_model = clf.best_estimator_
	elif(args.model=='decision tree'):
		#print(model.get_params().keys())
		parameters = {'max_depth':[4,6,8,12],
                      'min_samples_leaf':[5, 10, 20, 50, 100],
                      'min_samples_split': [2, 5, 10]}
		clf = GridSearchCV(model, parameters, cv=5)
		clf.fit(x_train, y_train)
		print('Best Regressor:', clf.best_estimator_.get_params())
		fitted_model = clf.best_estimator_
	elif(args.model=='linear'):
		fitted_model = model.fit(x_train, y_train)
	elif(args.model=='random forest'):
		#print(model.get_params().keys())
		parameters = {'bootstrap': [True],
                      'max_depth': [10, 50, 100],
                      'max_features': ['auto'],
                      #'min_samples_leaf': [1, 2, 4],
                      #'min_samples_split': [2, 5, 10],
                      'n_estimators': [200, 1000, 2000]}
		clf = GridSearchCV(model, parameters, cv=5)
		#clf = RandomizedSearchCV(model, parameters, n_iter=100, cv=5, random_state=42)
		clf.fit(x_train, y_train)
		print('Best Regressor:', clf.best_estimator_.get_params())
		fitted_model = clf.best_estimator_
	else:
		print('Wrong model selected!!!')
		fitted_model = []
	return fitted_model

def plot_preds_and_errors(predictions_legitimate, labels_legitimate, predictions_attack, labels_attack, index, simulation, args, title, scenario):
	#plt.figure()
	fig, axs = plt.subplots(2,2)
	axs[0][0].plot(labels_legitimate[index], color="green", label=r"$Actual$", linewidth=2)
	axs[0][0].plot(predictions_legitimate[index], color="blue", label=r"$Forecast$", linewidth=2)#, marker="o")
	axs[0][0].set_title('Legitimate')
	axs[0][0].legend()
	#plt.show()
	errors = []
	for pred, lab in zip(predictions_legitimate[index], labels_legitimate[index]):
		if(not (pred==0 and lab==0)):
			errors.append(pow(pred-lab, 2))#/pow(pred+lab,2))
		else:
			errors.append(0)
	#plt.figure()
	axs[1][0].plot(errors, color="red", label=r"$Error^{2}$", linewidth=2)
	#axs[1][0].set_title('Error')
	axs[1][0].legend()


	axs[0][1].plot(labels_attack[index], color="green", label=r"$Actual$", linewidth=2)
	axs[0][1].plot(predictions_attack[index], color="blue", label=r"$Forecast$", linewidth=2)#, marker="o")
	axs[0][1].set_title('{}'.format(scenario))
	axs[0][1].legend()
	#plt.show()
	errors = []
	for pred, lab in zip(predictions_attack[index], labels_attack[index]):
		if(not (pred==0 and lab==0)):
			errors.append(pow(pred-lab, 2))#/pow(pred+lab,2))
		else:
			errors.append(0)
	#plt.figure()
	axs[1][1].plot(errors, color="red", label=r"$Error^{2}$", linewidth=2)
	#axs[1][1].set_title('Error')
	axs[1][1].legend()

	fig.suptitle('{}'.format(args.features_to_fit[index]))
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	image_name = args.output_imgs_folder + scenario + '/' + simulation + '/' + title + '.png'
	#print(image_name)
	plt.savefig(image_name)
	plt.show()

def compare_legitimate_and_attack(legitimate_file, attack_file, attack, simulation, trained_predictors, args):
	#Legitimate
	inputs, labels = get_test_x_y(legitimate_file, args)
	inputs_legitimate, labels_legitimate = select_features_x_y(inputs, labels, args)
	if(args.standardization=='True'):
		inputs_legitimate, labels_legitimate = standardize_windows(inputs_legitimate, labels_legitimate, args)
	predictions_legitimate = []
	for index in range(len(inputs_legitimate)):
		predictions_legitimate.append(trained_predictors[index].predict(inputs_legitimate[index]))
		#plot_preds_and_errors(predictions, labels, index, simulation, args, title='Legitimate', scenario=attack)
	#Attack
	inputs, labels = get_test_x_y(attack_file, args)
	inputs_attack, labels_attack = select_features_x_y(inputs, labels, args)
	if(args.standardization=='True'):
		inputs_attack, labels_attack = standardize_windows(inputs_attack, labels_attack, args)
	predictions_attack = []
	for index in range(len(inputs_attack)):
		predictions_attack.append(trained_predictors[index].predict(inputs_attack[index]))
	
	#print(predictions_legitimate)
	for index in range(len(inputs_attack)):
		plot_preds_and_errors(predictions_legitimate, labels_legitimate, predictions_attack, labels_attack, index, simulation, args, title=attack, scenario=attack)

#New main function
def main():
	args = settings_parser.arg_parse()
	#Getting data path
	filenames = glob.glob(os.path.join(os.getcwd(), args.feat_folders, 'Legitimate','*'))
	filenames.sort()
	train_files, test_files = get_files_train_test(filenames, args)
	#print(train_files)

	x_train, y_train = get_x_y(train_files, 3, args)
	x_test, y_test = get_x_y(test_files, 1, args)

	print('Features to fit: {}'.format(args.features_to_fit))

	x_trains_list, y_trains_list = select_features_x_y(x_train, y_train, args)
	x_tests_list, y_tests_list = select_features_x_y(x_test, y_test, args)
	
	if(args.standardization=='True'):
		x_trains_list, y_trains_list = standardize_windows(x_trains_list, y_trains_list, args)
		x_tests_list, y_tests_list = standardize_windows(x_tests_list, y_tests_list, args)
	elif(args.normalization=='True'):
		x_trains_list, y_trains_list = normalize_windows(x_trains_list, y_trains_list, args)
		x_tests_list, y_tests_list = normalize_windows(x_tests_list, y_tests_list, args)

	trained_predictors = []
	#Train a predictor for each of the features that we want to fit and get corresponding error
	for index in range(len(x_trains_list)):
		#Plot distribution of training set and test set
		weights = np.ones_like(y_trains_list[index]) / len(y_trains_list[index])
		plt.hist(y_trains_list[index], weights=weights, color = 'blue', edgecolor = 'black')
		plt.xlabel("Feature values")
		plt.ylabel("Frequency")
		plt.title("Training data distribution - Feature: {}".format(args.features_to_fit[index]))
		plt.show()
		weights = np.ones_like(y_tests_list[index]) / len(y_tests_list[index])
		plt.hist(y_tests_list[index], weights=weights, color = 'blue', edgecolor = 'black')
		plt.xlabel("Feature values")
		plt.ylabel("Frequency")
		plt.title("Testing data distribution - Feature: {}".format(args.features_to_fit[index]))
		plt.show()

		#Define regressor
		regressor = define_model(args)
		print('Fitting regressor...')
		regressor = fit_model(regressor, x_trains_list[index], y_trains_list[index], args)
		trained_predictors.append(regressor)
		#Predict values for train set
		predictions = regressor.predict(x_trains_list[index])
		if(args.model!='linear'):
			predictions = np.expand_dims(predictions,axis=-1)
		indeces = y_trains_list[index]!=0
		#print(indeces.shape)
		pred = predictions[indeces]
		#print(pred)
		print('Feature: {}\t\tTrain Mean Squared Error: {}'.format(args.features_to_fit[index],metrics.mean_squared_error(y_trains_list[index], predictions)))
		plt.boxplot(pred, notch=True)
		plt.ylabel("Forecast values")
		plt.title("Training forecast box plot")
		plt.show()
		#Predict values for test set
		predictions = regressor.predict(x_tests_list[index])
		#print(predictions.shape)
		if(args.model!='linear'):
			predictions = np.expand_dims(predictions,axis=-1)
		indeces = y_tests_list[index]!=0
		pred = predictions[indeces]
		print('Feature: {}\t\tTest Mean Squared Error: {}'.format(args.features_to_fit[index],metrics.mean_squared_error(y_tests_list[index], predictions)))
		plt.boxplot(pred, notch=True)
		plt.ylabel("Forecast values")
		plt.title("Test forecast box plot")
		plt.show()

		#Scatter plot to check error with magnitude of feature
		xx = y_tests_list[index]
		print(y_tests_list[index].shape)
		print(predictions.shape)
		if(args.model!='linear'):
			yy = np.abs(y_tests_list[index] - predictions)#np.expand_dims(predictions,axis=-1))
		else:
			yy = np.abs(y_tests_list[index] - predictions)
		print(xx.shape)
		print(yy.shape)
		plt.scatter(xx, yy)
		plt.xlabel("True forecast")
		plt.ylabel("Forecase error")
		plt.title("Forecast error trend")
		plt.show()

	#Plot prediction on a single test file
	if(False):
		simulation='simulation-002'
		node='SENSOR-6'#'SINKNODE-17'
		attack='Clone_ID'
		#Pick Legitimate file
		legitimate_file = [args.feat_folders+'/Legitimate/'+simulation+'/'+node+'.csv']
		#Pick malicious file
		attack_file = [args.feat_folders+'/'+attack+'/'+simulation+'/'+node+'.csv']
		compare_legitimate_and_attack(legitimate_file, attack_file, attack, simulation, trained_predictors, args)
	
	#Get all squared errors of training set
	trains_pred = trained_predictors[index].predict(x_trains_list[index])
	if(args.model=='linear'):
		all_errors = np.power(trains_pred - y_trains_list[index], 2)
	else:
		all_errors = np.power(np.expand_dims(trains_pred,axis=-1) - y_trains_list[index], 2)
	print('All errors shape: {}'.format(all_errors.shape))
	print('All errors 99 percentile: {}'.format(np.percentile(all_errors, 99)))
	threshold = np.percentile(all_errors, 99)
	
	if(args.model=='linear'):
		print('Regressor weights: {} and bias: {}'.format(trained_predictors[index].coef_, trained_predictors[index].intercept_))
	#else:
	#	print('Regressor weights: {} and bias: {}'.format(trained_predictors[index].dual_coef_, trained_predictors[index].intercept_))
	
	#Try to detect anomalies
	simulation = 'simulation-001'
	#node='SENSOR-7'#'SINKNODE-17'
	nodes = ['SENSOR-1','SENSOR-2','SENSOR-3','SENSOR-4','SENSOR-5','SENSOR-6','SENSOR-7','SENSOR-8','SENSOR-9','SENSOR-10','SENSOR-11','SENSOR-12','SENSOR-13','SENSOR-14','SENSOR-15','SENSOR-16','SINKNODE-17']
	attack='Legitimate'
	attack_time = 215
	attack_window_start = (int)((attack_time-args.time_start)/args.time_window) - args.lag_val
	for node in nodes:
		attack_file = [args.feat_folders+'/'+attack+'/'+simulation+'/'+node+'.csv']
		#print(attack_file)
		inputs, labels = get_x_y(attack_file, 1, args)
		inputs_attack, labels_attack = select_features_x_y(inputs, labels, args)

		if(args.standardization=='True'):
			inputs_attack, labels_attack = standardize_windows(inputs_attack, labels_attack, args)
		elif(args.normalization=='True'):
			inputs_attack, labels_attack = normalize_windows(inputs_attack, labels_attack, args)
	
		predictions_attack = []
		predictions_attack.append(trained_predictors[index].predict(inputs_attack[index]))
		if(args.model=='linear'):
			test_errors = np.power(predictions_attack[index] - labels_attack[index], 2)
		else:
			test_errors = np.power(np.expand_dims(predictions_attack[index],axis=-1) - labels_attack[index], 2)
		anomalies = test_errors[test_errors>threshold]
		anomalies_indeces = (test_errors>threshold).tolist()
		anomalies_indeces = [i for i, x in enumerate(anomalies_indeces) if x == [True]]
		fig, axs = plt.subplots(2,1)
		axs[0].plot(predictions_attack[index], label='Forecast')
		axs[0].plot(labels_attack[index], label='Label')
		axs[0].axvline(x=attack_window_start, label='Attack start', color='red', linestyle='dotted', alpha=0.5)
		axs[0].set_xlabel('Time windows (10s)')
		axs[0].set_ylabel('Standardized metric')
		axs[0].legend()
		axs[1].plot(test_errors, label='Error', color='yellowgreen')
		axs[1].plot(anomalies_indeces, test_errors[anomalies_indeces], 'rx', label='Anomalies Detected')
		axs[1].set_xlabel('Time windows (10s)')
		axs[1].set_ylabel('Error')
		axs[1].legend()
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])
		plt.suptitle(attack.replace("_", " ") + ' - ' + node)
		#plt.savefig(attack.replace("_", " ") + ' - ' + node)
		plt.show()

if __name__ == '__main__':
	main()



#Old code not used anymore but that can might be useful in future

'''
#Get x and y from all csv files used in training
def get_train_x_y(train_files, args):
	file = train_files[0]
	x_train = []
	y_train = []	
	#Split training files in non-overlapping series
	for file in train_files:
		data = pd.read_csv(file)
		for index in range(0, len(data.index), args.lag_val + args.prediction_window):
			if(index+args.lag_val + args.prediction_window < len(data.index)):
				x_train.append(np.asarray(data.iloc[index:index+args.lag_val]))
				y_train.append(np.asarray(data.iloc[index+args.lag_val:index+args.lag_val+args.prediction_window]))
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	#x_train and y_train: n_series x n_elements_per_series x n_features
	#x_train: n_series x val_lag x n_features
	#x_test: n_series x prediction_window x n_features
	print('x_train shape: {}\ty_train shape: {}'.format(x_train.shape, y_train.shape))
	#print(np.asarray(x_train)[:,:,0])
	#print(np.asarray(y_train)[:,:,0])
	return x_train, y_train

def get_test_x_y(test_files, args):
	x_test = []
	y_test = []
	#Split training files with overlapping series like for real-time predictions
	for file in test_files:
		data = pd.read_csv(file)
		for index in range(0, len(data.index) - (args.lag_val + args.prediction_window) + 1):
			if(index+args.lag_val + args.prediction_window < len(data.index)):
				x_test.append(np.asarray(data.iloc[index:index+args.lag_val]))
				y_test.append(np.asarray(data.iloc[index+args.lag_val:index+args.lag_val+args.prediction_window]))
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)
	#x_test and y_test: n_series x n_elements_per_series x n_features
	#x_test: n_series x val_lag x n_features
	#y_test: n_series x prediction_window x n_features
	#print('x_test shape: {}\ty_test shape: {}'.format(x_test.shape, y_test.shape))
	#print(np.asarray(x_test)[:,:,0])
	#print(np.asarray(y_test)[:,:,0])
	return x_test, y_test
'''