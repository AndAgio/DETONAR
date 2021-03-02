#Python modules
import pandas as pd
import numpy as np
import os
import glob
import torch
import warnings
warnings.filterwarnings("ignore")
#Python files
import settings_parser

def extract_data_from_csv(path_to_file):
	#print('Reading csv file...')
	data = pd.read_csv(path_to_file)
	#print('Removing columns full of NaN...')
	#print('List of columns before\n{}'.format(data.columns))
	for column in data.columns:
		if (data[column].isnull().all() and column!='SEGMENT_LEN'):
			#print('Removing column named {}'.format(column))
			data = data.drop(column, axis=1)
	#print('List of columns after\n{}'.format(data.columns))

	return data

def get_flows_from_data(data):
	#print('Getting unique node\'s names...')
	nodes_names = data['TRANSMITTER_ID'].unique()
	nodes_names = [i for i in nodes_names if ('SENSOR' in i or 'SINKNODE' in i)]
	#print(nodes_names)

	#print('Getting flow of packets for each node...')
	flows = []
	for name in nodes_names:
		flows.append(data[data['TRANSMITTER_ID']==name])
	#print('A flow: {}'.format(flows[0]))
	#print('flows length: {}'.format(len(flows)))
	#for flow in flows:
	#	print('flow length: {}'.format(len(flow)))
	return nodes_names, flows

def column_translation(column):
	unique_values_list = column.unique()
	list_of_indeces = range(1, len(unique_values_list)+1)
	zip_dict = zip(unique_values_list, list_of_indeces)
	dict_trans = dict(zip_dict)
	column = column.replace(dict_trans, inplace=False)
	return column


def refine_flows(flows):
	#print('Refining flows for string variables and columns with NaN values...')
	for i in range(len(flows)):
		#flows[i] = flows[i].fillna(0.0)
		for column in flows[i].columns:
			#print('Column: {} -> number type: {}'.format(column,np.issubdtype(flows[i][column].dtype, np.number)))
			if(not(np.issubdtype(flows[i][column].dtype, np.number))): #or flows[i][column].isnull().any()):
				#print(column)
				#flows[i] = pd.concat([flows[i],pd.get_dummies(flows[i][column], prefix=column, dummy_na=True)],axis=1).drop([column],axis=1)
				flows[i][column] = column_translation(flows[i][column])
			flows[i].fillna(0, inplace= True)
			if('(US)' in column):
				flows[i][column] = flows[i][column]/1e6

	return flows

def get_sequences_from_flows(args, flows, mode):
	#print('Getting sequences from flows...')
	#Getting parameters necessary to build a sequence
	pcks_p_seq = args.n_packets_in_sequence
	pcks_size = len(flows[0].columns) - 1 
	#print(flows[0].columns)
	#Getting parameters to creating the dataset of all sequences
	n_flows = len(flows)
	if(mode=='train'):
		seq_p_flow = args.n_train_sequences_in_flow
	else:
		seq_p_flow = args.n_valid_sequences_in_flow
	#Building sequences structure
	sequences = np.zeros((n_flows*seq_p_flow, pcks_p_seq, pcks_size))
	#Picking random sequences
	index = 0
	for i in range(n_flows):
		for j in range(seq_p_flow):
			start = random.randint(0, len(flows[i])-pcks_p_seq-1)
			end = start + pcks_p_seq
			selected_rows = flows[i].iloc[start:end]
			sequences[index,:,:] = (selected_rows.drop('PCKT_LABEL',axis=1)).to_numpy()
			index += 1
	return sequences

def get_dataset(mode):
	#Importing parser arguments
	args = settings_parser.arg_parse()
	#Getting data path
	filenames = glob.glob(os.path.join(os.getcwd(), '..', args.data_dir, '*.csv'))
	filenames.sort()
	all_sequences = []
	index = 0
	#print("Processing csv files")
	for filename in filenames:
		name =  filename.split("/")[-1]
		print("\rReading file: {}\t{}/{}".format(name, index+1, len(filenames)), end="\r")
		if(index == len(filenames)-1):
			print()
		path = os.path.join(os.getcwd(), '..', args.data_dir, filename)
		#Proceding with extraction of sequences from csv file
		data = extract_data_from_csv(path)
		nodes_ids, flows = get_flows_from_data(data)
		flows = refine_flows(flows)
		sequences = get_sequences_from_flows(args, flows, mode)
		if (index==0):
			all_sequences = sequences
		else:
			#print('all_sequences.shape: {} and sequences.shape: {}'.format(all_sequences.shape, sequences.shape))
			all_sequences = np.concatenate((all_sequences, sequences), axis=0)
		index += 1
	return all_sequences

if __name__ == '__main__':
	print("Importing training set")
	trainset = get_dataset('train')
	print('Train-set shape: {}'.format(trainset.shape))￼
	print("\nImporting validation set")
	validset = get_dataset('valid')
	print('Validation-set shape: {}'.format(validset.shape))