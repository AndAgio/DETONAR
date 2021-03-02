# Python modules
import pandas as pd
import numpy as np
import os
import glob
import networkx as nx
import matplotlib.pyplot as plt
import time as tm
# Python files
import settings_parser

''' #Old extract_data_up_to function
def extract_data_up_to(data, time):
	#From the pandas dataframe extract only those packets arrived up to a certain second
	condition = (data['NW_LAYER_ARRIVAL_TIME(US)']<=time*1e6)
	data = data[condition]
	#Remove all columns that should not be considered to compute the DODAG
	for column in data.columns:
		if (column!='CONTROL_PACKET_TYPE/APP_NAME' and column!='SOURCE_ID' and column!='DESTINATION_ID'):
			data = data.drop(column, axis=1)
	#Keep only the DAO packets which are the only one used to build the DODAG
	data = data[data['CONTROL_PACKET_TYPE/APP_NAME']=='DAO']
	return data
'''


def extract_data_up_to(data, time, args):
    # From the pandas dataframe extract only those packets arrived up to a certain second
    # condition = (data['NW_LAYER_ARRIVAL_TIME(S)']<=time)
    condition = (data[args.time_feat_sec] <= time)
    data = data[condition]
    return data


def refine_edges_list(edges):
    edges_to_remove = list()
    # For each edges in the list of edges check if two edges have the same source
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            source_node_first_link = edges[i][0]
            source_node_second_link = edges[j][0]
            # If they have the same source then the oldest link must be removed since it means that the node had changed parent
            if (source_node_first_link == source_node_second_link):
                edges_to_remove.append(edges[i])
    # Get a set from the list of edges to remove and remove all of them from the original edges list
    edges_to_remove = set(edges_to_remove)
    for ed in edges_to_remove:
        edges.remove(ed)

    return edges


''' #Old get dodag function
def get_dodag(data):
	# Create the list for nodes and edges and the corresponding empty graph
	list_of_nodes_already_added = []
	list_of_edges_already_added = []
	dodag = nx.Graph()
	#For each DAO packet in the dataframe...
	for index, row in data.iterrows():
		#Get IDs as numbers
		source_id = row['SOURCE_ID'].split("-")[-1]
		destination_id = row['DESTINATION_ID'].split("-")[-1]
		#Check if the device is already in the graph list of nodes, if not add it
		if(source_id not in list_of_nodes_already_added):
			list_of_nodes_already_added.append(source_id)		
		if(destination_id not in list_of_nodes_already_added):
			list_of_nodes_already_added.append(destination_id)
		#Get the edge between the two nodes involved in DAO communication
		considered_edge = (source_id, destination_id)
		#Check id this edge is already in the graph list of edges, if not add it
		if(considered_edge not in list_of_edges_already_added):
			list_of_edges_already_added.append(considered_edge)

	#Refine the list of edges in order to keep only most recent father-son relationships
	list_of_edges_already_added = refine_edges_list(list_of_edges_already_added)
	#Build the DODAG graph from nodes and edges lists
	dodag.add_nodes_from(list_of_nodes_already_added)
	dodag.add_edges_from(list_of_edges_already_added)

	return dodag
'''


def get_dodag(data):
    # tic = tm.perf_counter()
    # Get source ids and dest ids only with numbers
    source_ids = data['SOURCE_ID'].values.tolist()
    source_ids = [id.split("-")[-1] for id in source_ids]
    dest_ids = data['DESTINATION_ID'].values.tolist()
    dest_ids = [id.split("-")[-1] for id in dest_ids]
    # Each DAO represents a potential edge
    edges = [(source_ids[i], dest_ids[i]) for i in range(len(source_ids))]
    # Remove duplicate potential edges and maintain the order
    seen = set()
    seen_add = seen.add
    edges = [x for x in edges if not (x in seen or seen_add(x))]
    # Get list of nodes names (order doesn't matter)
    source_ids = list(dict.fromkeys(source_ids))
    dest_ids = list(dict.fromkeys(dest_ids))
    list_of_nodes = source_ids + dest_ids
    list_of_nodes = list(dict.fromkeys(list_of_nodes))
    # Use nx to obtain the graph corresponding to the dodag
    dodag = nx.Graph()
    # Refine the list of edges in order to keep only most recent father-son relationships
    edges = refine_edges_list(edges)
    # Build the DODAG graph from nodes and edges lists
    dodag.add_nodes_from(list_of_nodes)
    dodag.add_edges_from(edges)
    # toc = tm.perf_counter()
    # print('New get DODAG takes: {}'.format(toc-tic))

    return dodag


def extract_dodag_before_after(data, list_nodes, neighbors, time, args):
    # print('Extracting DODAG before and after the anomaly...')
    tic = tm.perf_counter()
    # Get DODAG some time steps before the anomaly is raised
    window_time = args.time_window
    data_before = extract_data_up_to(data, time - 1 * window_time, args)
    dodag_before = get_dodag(data_before)
    # Plot the graph corresponding to the DODAG extracted
    # fig, axs = plt.subplots(1, 3, figsize=(30,15))
    # axs[0].set_title('DODAG before anomaly')
    # nx.draw(dodag_before, with_labels=True, ax=axs[0])
    # Get DODAG after the anomaly is raised
    data_after = extract_data_up_to(data, time, args)
    dodag_after = get_dodag(data_after)
    # Plot the graph corresponding to the DODAG extracted
    # pos = nx.spring_layout(dodag_after)
    # axs[1].set_title('DODAG after anomaly')
    # nx.draw(dodag_after, pos=pos, with_labels=True, ax=axs[1])
    # nx.draw(dodag_after.subgraph(list_nodes), pos=pos, node_color='orange', with_labels=True, ax=axs[1])
    # nx.draw(dodag_after.subgraph(neighbors), pos=pos, node_color='yellow', with_labels=True, ax=axs[1])
    # Compute difference between graphs and plot everything
    dodag_difference = nx.difference(dodag_after, dodag_before)
    dodag_difference.remove_nodes_from(list(nx.isolates(dodag_difference)))
    # axs[2].set_title('DODAG difference')
    # nx.draw(dodag_difference, with_labels=True, ax=axs[2])
    # plt.suptitle(args.scenario)
    toc = tm.perf_counter()
    # print('Everything DODAG took {:.5f}'.format(toc - tic))
    # plt.show()
    if (len(dodag_difference) == 0):
        return False, []
    nodes_changing = dodag_difference.nodes()
    # print('Nodes changing: {}'.format(nodes_changing))
    return True, nodes_changing


def main():
    # Just a trial to check if the code works, the main shouldn't be used actually
    args = settings_parser.arg_parse()
    window_time = 10
    time_step = 25
    time_anomaly = time_step * 10 + 10
    extract_dodag_before_after(os.path.join(os.getcwd(), '..', args.data_dir, 'Sinkhole', 'Packet_Trace_600s/001.csv'),
                               time_anomaly, window_time)


if __name__ == '__main__':
    main()

'''
def extract_data_from_csv(path_to_file):
	print('Reading csv file...')
	data = pd.read_csv(path_to_file)
	for column in data.columns:
		if (column!='CONTROL_PACKET_TYPE/APP_NAME' and column!='SOURCE_ID' and column!='DESTINATION_ID'):
			data = data.drop(column, axis=1)
	#print('List of columns after\n{}'.format(data.columns))
	data = data[data['CONTROL_PACKET_TYPE/APP_NAME']=='DAO']
	print(data)
	return data

def extract_data_from_csv_up_to(path_to_file, time):
	print('Reading csv file...')
	data = pd.read_csv(path_to_file, index_col=False)
	#print(data[data['CONTROL_PACKET_TYPE/APP_NAME']=='DAO'])
	#print(time*1e6)
	condition = (data['NW_LAYER_ARRIVAL_TIME(US)']<=time*1e6)
	data = data[condition]
	#print(data)
	#print(data[data['CONTROL_PACKET_TYPE/APP_NAME']=='DAO'])
	for column in data.columns:
		if (column!='CONTROL_PACKET_TYPE/APP_NAME' and column!='SOURCE_ID' and column!='DESTINATION_ID'):
			data = data.drop(column, axis=1)
	#print('List of columns after\n{}'.format(data.columns))
	data = data[data['CONTROL_PACKET_TYPE/APP_NAME']=='DAO']
	#print(data)
	return data
'''
