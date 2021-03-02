# Python modules
import pandas as pd
import numpy as np
import math
import os
import glob
import random
import warnings
import multiprocessing as mp
import time as time
import concurrent.futures
import itertools

warnings.filterwarnings("ignore")
# Python files
import settings_parser


def get_data(path_to_file):
    # Read csv file
    data = pd.read_csv(path_to_file, index_col=False)
    # For each column check if it contains a time value in micro seconds, if so bring it to seconds
    for column in data.columns:
        if ('(US)' in column):
            # Sometimes time values are cut due to NetSim simulator so we need to replace them with nan values
            if (data[column].dtype == object):
                data[column] = data[column].replace('N', np.nan, regex=True)
                data[column] = data[column].replace('Na', np.nan, regex=True)
                data[column] = data[column].replace('', np.nan, regex=True)
                data[column] = pd.to_numeric(data[column])
            data[column] = data[column] / 1e6
    return data


def get_unique_nodes_names(data):
    # Get names of transmitter devices
    nodes_names = data['TRANSMITTER_ID'].unique()
    # Remove nan values
    nodes_names = [i for i in nodes_names if str(i) != 'nan']
    # Remove nodes that are not sensors
    nodes_names = [i for i in nodes_names if ('SENSOR' in i or 'SINKNODE' in i)]
    return nodes_names


def get_time_window_data(data, index, args, full_data=False):
    time_window = args.time_window
    if (full_data):
        start_time = args.time_start
    else:
        start_time = index * time_window + args.time_start
    end_time = (index + 1) * time_window + args.time_start
    # Get all packets that have been received at the network layer between start and end time (depending on window size)
    # condition = (data['NW_LAYER_ARRIVAL_TIME(US)']>start_time) & (data['NW_LAYER_ARRIVAL_TIME(US)']<=end_time)
    condition = (data[args.time_feat_micro] > start_time) & (data[args.time_feat_micro] <= end_time)
    sequence = data[condition]
    # print(sequence)
    return sequence


def get_features(data, node_name, args):  # full_data, node_name, args):
    # print("Node name: {}".format(node_name))
    features = np.zeros((args.n_features), dtype=float)
    # Get received and sent packets
    received_packets = data[data['RECEIVER_ID'] == node_name]
    transmitted_packets = data[data['TRANSMITTER_ID'] == node_name]
    # print(received_packets)
    # Get number of DIO received
    received_count = received_packets['CONTROL_PACKET_TYPE/APP_NAME'].value_counts()
    # print('\nReceived count:\n{}'.format(received_count))
    if ('DIO' in received_count):
        features[0] = received_count['DIO']
    # Get number of DIO transmitted
    transmitted_count = transmitted_packets['CONTROL_PACKET_TYPE/APP_NAME'].value_counts()
    # print('\nTransmitted count:\n{}'.format(transmitted_count))
    if ('DIO' in transmitted_count):
        features[1] = transmitted_count['DIO']

    # Get number of DAO received
    if ('DAO' in received_count):
        features[2] = received_count['DAO']
    # Get number of DAO transmitted
    if ('DAO' in transmitted_count):
        features[3] = transmitted_count['DAO']

    # Get number of DIS received
    if ('DIS' in received_count):
        features[4] = received_count['DIS']
    # Get number of DAO transmitted
    if ('DIS' in transmitted_count):
        features[5] = transmitted_count['DIS']

    # Get number of application packets received
    received_app_count = received_packets['PACKET_TYPE'].value_counts()
    if ('Control_Packet' in received_app_count):
        features[6] = received_app_count.sum() - received_app_count['Control_Packet']
    else:
        features[6] = received_app_count.sum()
    # Get number of application packets transmitted
    transmitted_app_count = transmitted_packets['PACKET_TYPE'].value_counts()
    if ('Control_Packet' in transmitted_app_count):
        features[7] = transmitted_app_count.sum() - transmitted_app_count['Control_Packet']
    else:
        features[7] = transmitted_app_count.sum()

    # Get all packets received/transmitted by the node
    all_packets = data[(data['RECEIVER_ID'] == node_name) | (data['TRANSMITTER_ID'] == node_name)]
    app_list = all_packets['CONTROL_PACKET_TYPE/APP_NAME'].value_counts().index.to_list()
    control_list = ['DIO', 'DAO', 'DIS', 'DAO-ACK', 'OSPF_HELLO',
                    'OSPF_DD', 'OSPF_LSREQ', 'OSPF_LSUPDATE', 'OSPF_LSACK']
    for control in control_list:
        if (control in app_list):
            app_list.remove(control)
    features[8] = len(app_list)

    # Get number of different source IPs
    source_ips = all_packets['SOURCE_IP'].value_counts().index.to_list()
    features[9] = len(source_ips)

    # Get number of different source IPs
    destination_ips = all_packets['DESTINATION_IP'].value_counts().index.to_list()
    features[10] = len(destination_ips)

    # Get number of Gateway IPs
    gateway_ips = transmitted_packets['GATEWAY_IP'].value_counts().index.to_list()
    features[11] = len(gateway_ips)

    # Get successful transmission rate
    successful = transmitted_packets[transmitted_packets['PACKET_STATUS'] == 'Successful']
    collided = transmitted_packets[transmitted_packets['PACKET_STATUS'] == 'Collided']
    if (len(transmitted_packets.index) == 0):
        rate = 100
    else:
        rate = len(successful.index) / len(transmitted_packets.index) * 100
    features[12] = rate

    # Get number of broadcast packets sent
    broadcast = transmitted_packets[transmitted_packets['DESTINATION_IP'] == 'FF00:0:0:0:0:0:0:0']
    features[13] = len(broadcast.index)

    '''
    #Get neighbours (number of different nodes who received DIO messages sent by this node)
    transmitted_dios = transmitted_packets[transmitted_packets['CONTROL_PACKET_TYPE/APP_NAME'] == 'DIO']
    neighbours = transmitted_dios['RECEIVER_ID'].value_counts()
    features[14] = len(neighbours.index)

    #Get number of nexthop IP
    all_transmitted_packets = full_data[full_data['TRANSMITTER_ID'] == node_name]
    next_hop_ips = all_transmitted_packets[all_transmitted_packets['NEXT_HOP_IP'] != 'FF00:0:0:0:0:0:0:0']['NEXT_HOP_IP'].value_counts().index.to_list()
    features[15] = len(next_hop_ips)
    #print('next_hop_ips: {}'.format(len(next_hop_ips)))
    '''

    # Get number of incoming application packets that do not have itself as destination
    received_app_pcks = received_packets[received_packets['PACKET_TYPE'] == 'Sensing']
    # print('received_app_pcks: {}'.format(received_app_pcks))
    incoming = received_app_pcks[(received_app_pcks['DESTINATION_ID'] != received_app_pcks['RECEIVER_ID']) & (received_app_pcks['PACKET_STATUS'] == 'Successful')]
    n_incoming_app_pcks = len(incoming.index)
    # print('n_incoming_app_pcks: {}'.format(n_incoming_app_pcks))
    # Get number of outgoing application packets that do not have itself as source
    transmitted_app_pcks = transmitted_packets[transmitted_packets['PACKET_TYPE'] == 'Sensing']
    # print('received_app_pcks: {}'.format(received_app_pcks))
    outgoing = transmitted_app_pcks[
        transmitted_app_pcks['SOURCE_ID'] != transmitted_app_pcks['TRANSMITTER_ID']]
    n_outgoing_app_pcks = len(outgoing.index)
    # print('n_outgoing_app_pcks: {}'.format(n_outgoing_app_pcks))
    if (n_incoming_app_pcks == n_outgoing_app_pcks):
        features[14] = 1#00
    elif (n_outgoing_app_pcks != 0 and n_incoming_app_pcks == 0):
        features[14] = 1#00
    else:
        features[14] = n_outgoing_app_pcks / n_incoming_app_pcks #* 100

    # Get number of nodes transmitting packets in this time window
    # list_nodes = data['TRANSMITTER_ID'].value_counts().index.to_list()
    # list_nodes = [node for node in list_nodes if 'SENSOR' in node]
    # print('list_nodes: {}'.format(list_nodes))
    # features[17] = list_nodes

    return features


def main():
    tic = time.time()
    args = settings_parser.arg_parse()
    # Getting files depending on data directory, scenario and simulation time chosen
    filenames = glob.glob(os.path.join(os.getcwd(), '..', args.data_dir, args.scenario,
                                       'Packet_Trace_' + str(int(args.simulation_time)) + 's', '*.csv'))
    filenames.sort()
    # print(filenames)
    ###################################################################################
    # This part is for debug only while trying out with parallelization
    if (args.single_sim != ''):
        filenames = [filename for filename in filenames if args.single_sim in filename]
    # print(filenames)
    ##################################################################################
    print('Extracting features for {} scenario'.format(args.scenario))
    # Cycle through each file in the scenario
    for file_index in range(len(filenames)):
        # Extract simulation name
        name = filenames[file_index].split("/")[-1]
        print("\rReading file: {}\t{}/{}".format(name, file_index + 1, len(filenames)), end="\r")
        if (file_index == len(filenames) - 1):
            print()
        # Extract data from the considered simulation
        data = get_data(filenames[file_index])
        # Get names of nodes inside a simulation
        nodes_names = get_unique_nodes_names(data)
        # For each node in the simulation
        for node_index in range(len(nodes_names)):
            # Get index for windowing the network traffic
            start_index = 0
            end_index = math.floor((args.simulation_time - args.time_start) / args.time_window)
            matrix = []
            # For each index get the corresponding network traffic window and extract the features in that window
            for index in range(start_index, end_index):
                time_sequence = get_time_window_data(data, index, args, full_data=False)
                # time_sequence_from_0_to_index = get_time_window_data(data, index, args, full_data=True)
                features = get_features(time_sequence, nodes_names[node_index],
                                        args)  # time_sequence_from_0_to_index, nodes_names[node_index], args)
                # Append the features row to the matrix that will contain the series of all features
                matrix.append(features)
            matrix = np.asarray(matrix)
            # If the cumulative features are considered then we need to compute the cumulative sum for each feature through time
            if (args.cumulative_sum == 'True'):
                # Cumulative sum over the lag window
                w = args.lag_val
                matrix = np.cumsum(matrix, axis=0)
            # Convert the matrix into a pandas dataframe and set features names
            pandas_matrix = pd.DataFrame(matrix)
            features_list = ['# DIO rcvd', '# DIO txd', '# DAO rcvd', '# DAO txd', '# DIS rcvd',
                             '# DIS txd', '# APP rcvd', '# APP txd', '# different APPs',
                             '# source IPs', '# dest IPs', '# gateway IPs', 'Succ rate', '# broadcasted',
                             'incoming_vs_outgoing']  # '# neighbours', '# next-hop IPs', 'incoming_vs_outgoing']
            pandas_matrix.columns = features_list
            # Save the csv file containing features for a single device
            name = args.out_feat_files + '/' + args.scenario + '/simulation-' + \
                   filenames[file_index].split("/")[-1].split("\\")[-1].split(".")[0] + \
                   '/' + nodes_names[node_index] + '.csv'
            # Create folder of simulation under the corresponding scenario in the folder containing features
            if not os.path.exists('/'.join(name.split("/")[:-1])):
                os.makedirs('/'.join(name.split("/")[:-1]))
            pandas_matrix.to_csv(name, index=False, header=True)
    toc = time.time()
    print('Time taken without parallelization: {}'.format(toc - tic))


# Try using parallelization to extract features
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def parallelized_main():
    pool = mp.Pool(mp.cpu_count())
    tic = time.time()
    args = settings_parser.arg_parse()
    # Getting files depending on data directory, scenario and simulation time chosen
    filenames = glob.glob(os.path.join(os.getcwd(), '..', args.data_dir, args.scenario,
                                       'Packet_Trace_' + str(int(args.simulation_time)) + 's', '*.csv'))
    filenames.sort()
    ###################################################################################
    # This part is for debug only while trying out with parallelization
    if (args.single_sim != ''):
        filenames = [filename for filename in filenames if args.single_sim in filename]
    # print(filenames)
    ##################################################################################
    print('Extracting features for {} scenario'.format(args.scenario))
    # Cycle through each file in the scenario
    for file_index in range(len(filenames)):
        # Extract simulation name
        name = filenames[file_index].split("/")[-1]
        print("\rReading file: {}\t{}/{}".format(name, file_index + 1, len(filenames)), end="\r")
        if (file_index == len(filenames) - 1):
            print()
        # Extract data from the considered simulation
        data = get_data(filenames[file_index])
        # Get names of nodes inside a simulation
        nodes_names = get_unique_nodes_names(data)
        # Create list containing every time window
        start_index = 0
        end_index = math.floor((args.simulation_time - args.time_start) / args.time_window)
        windows = []
        # For each index get the corresponding network traffic window
        for index in range(start_index, end_index):
            time_sequence = get_time_window_data(data, index, args, full_data=False)
            windows.append(time_sequence)
        # print('Windows length: {}'.format(len(windows)))
        # For each node in the simulation take all time windows and extract the features
        for node_index in range(len(nodes_names)):
            matrix = []
            # For each index get the corresponding network traffic window and extract the features in that window
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for features in executor.map(lambda x, y, z: get_features(x, y, z), windows,
                                             itertools.repeat(nodes_names[node_index], len(windows)),
                                             itertools.repeat(args, len(windows))):
                    matrix.append(features)
                    # print(matrix)
            '''
            for index in range(len(windows)):
                features = get_features(windows[index], nodes_names[node_index], args)
                #Append the features row to the matrix that will contain the series of all features
                matrix.append(features)
            '''
            matrix = np.asarray(matrix)
            # If the cumulative features are considered then we need to compute the cumulative sum for each feature through time
            if (args.cumulative_sum == 'True'):
                # Cumulative sum over the lag window
                w = args.lag_val
                matrix = np.cumsum(matrix, axis=0)
            # Convert the matrix into a pandas dataframe and set features names
            pandas_matrix = pd.DataFrame(matrix)
            features_list = ['# DIO rcvd', '# DIO txd', '# DAO rcvd', '# DAO txd', '# DIS rcvd',
                             '# DIS txd', '# APP rcvd', '# APP txd', '# different APPs',
                             '# source IPs', '# dest IPs', '# gateway IPs', 'Succ rate', '# broadcasted',
                             'incoming_vs_outgoing']  # '# neighbours', '# next-hop IPs', 'incoming_vs_outgoing']
            pandas_matrix.columns = features_list
            # Save the csv file containing features for a single device
            name = args.out_feat_files + '/' + args.scenario + '/simulation-' + \
                   filenames[file_index].split("/")[-1].split("\\")[-1].split(".")[0] + \
                   '/' + nodes_names[node_index] + '.csv'
            # Create folder of simulation under the corresponding scenario in the folder containing features
            if not os.path.exists('/'.join(name.split("/")[:-1])):
                os.makedirs('/'.join(name.split("/")[:-1]))
            pandas_matrix.to_csv(name, index=False, header=True)

        # Storing file with DAOs only
        # print('Storing DAOs csv')
        dao_packets = data[data['CONTROL_PACKET_TYPE/APP_NAME'] == 'DAO']
        #dao_packets = dao_packets[['SOURCE_ID', 'DESTINATION_ID', 'NW_LAYER_ARRIVAL_TIME(US)']]
        dao_packets = dao_packets[['SOURCE_ID', 'DESTINATION_ID', args.time_feat_micro]]
        # print(dao_packets['NW_LAYER_ARRIVAL_TIME(US)'])
        #dao_packets = dao_packets.rename(columns={'NW_LAYER_ARRIVAL_TIME(US)': 'NW_LAYER_ARRIVAL_TIME(S)'})
        dao_packets = dao_packets.rename(columns={args.time_feat_micro: args.time_feat_sec})
        name = args.out_feat_files + '/' + args.scenario + '/simulation-' + \
               filenames[file_index].split("/")[-1].split("\\")[-1].split(".")[0] + \
               '/' + 'DAOs.csv'
        # Create folder of simulation under the corresponding scenario in the folder containing features
        if not os.path.exists('/'.join(name.split("/")[:-1])):
            os.makedirs('/'.join(name.split("/")[:-1]))
        dao_packets.to_csv(name, index=False, header=True)

        # Storing file with DIOs only
        # print('Storing DIOs csv')
        dio_packets = data[data['CONTROL_PACKET_TYPE/APP_NAME'] == 'DIO']
        #dio_packets = dio_packets[['TRANSMITTER_ID', 'RECEIVER_ID', 'NW_LAYER_ARRIVAL_TIME(US)']]
        #dio_packets = dio_packets.rename(columns={'NW_LAYER_ARRIVAL_TIME(US)': 'NW_LAYER_ARRIVAL_TIME(S)'})
        dio_packets = dio_packets[['TRANSMITTER_ID', 'RECEIVER_ID', args.time_feat_micro]]
        dio_packets = dio_packets.rename(columns={args.time_feat_micro: args.time_feat_sec})
        name = args.out_feat_files + '/' + args.scenario + '/simulation-' + \
               filenames[file_index].split("/")[-1].split("\\")[-1].split(".")[0] + \
               '/' + 'DIOs.csv'
        # Create folder of simulation under the corresponding scenario in the folder containing features
        if not os.path.exists('/'.join(name.split("/")[:-1])):
            os.makedirs('/'.join(name.split("/")[:-1]))
        dio_packets.to_csv(name, index=False, header=True)

        # Storing file with Sensing packets only
        # print('Storing APPs csv')
        app_packets = data[data['PACKET_TYPE'] == 'Sensing']
        #app_packets = app_packets[['TRANSMITTER_ID', 'RECEIVER_ID', 'NW_LAYER_ARRIVAL_TIME(US)']]
        #app_packets = app_packets.rename(columns={'NW_LAYER_ARRIVAL_TIME(US)': 'NW_LAYER_ARRIVAL_TIME(S)'})
        app_packets = app_packets[['TRANSMITTER_ID', 'RECEIVER_ID', args.time_feat_micro]]
        app_packets = app_packets.rename(columns={args.time_feat_micro: args.time_feat_sec})
        name = args.out_feat_files + '/' + args.scenario + '/simulation-' + \
               filenames[file_index].split("/")[-1].split("\\")[-1].split(".")[0] + \
               '/' + 'APPs.csv'
        # Create folder of simulation under the corresponding scenario in the folder containing features
        if not os.path.exists('/'.join(name.split("/")[:-1])):
            os.makedirs('/'.join(name.split("/")[:-1]))
        app_packets.to_csv(name, index=False, header=True)

        # Storing file with any packet but transmitter and time only
        # print('Storing tx_time csv')
        #all_packets = data[['TRANSMITTER_ID', 'NW_LAYER_ARRIVAL_TIME(US)', 'NEXT_HOP_IP']]
        #all_packets = all_packets.rename(columns={'NW_LAYER_ARRIVAL_TIME(US)': 'NW_LAYER_ARRIVAL_TIME(S)'})
        all_packets = data[['TRANSMITTER_ID', args.time_feat_micro, 'NEXT_HOP_IP']]
        all_packets = all_packets.rename(columns={args.time_feat_micro: args.time_feat_sec})
        name = args.out_feat_files + '/' + args.scenario + '/simulation-' + \
               filenames[file_index].split("/")[-1].split("\\")[-1].split(".")[0] + \
               '/' + 'ALL_tx_time.csv'
        # Create folder of simulation under the corresponding scenario in the folder containing features
        if not os.path.exists('/'.join(name.split("/")[:-1])):
            os.makedirs('/'.join(name.split("/")[:-1]))
        all_packets.to_csv(name, index=False, header=True)

        # Storing file with control packets and ranks & versions
        # print('Storing RANKs csv')
        ranks_vers = data[data['PACKET_TYPE'] == 'Control_Packet']
        #ranks_vers = ranks_vers[['TRANSMITTER_ID', 'CONTROL_PACKET_TYPE/APP_NAME', 'NW_LAYER_ARRIVAL_TIME(US)', 'RPL_RANK', 'RPL_VERSION']]
        #ranks_vers = ranks_vers.rename(columns={'NW_LAYER_ARRIVAL_TIME(US)': 'NW_LAYER_ARRIVAL_TIME(S)'})
        ranks_vers = ranks_vers[['TRANSMITTER_ID', 'CONTROL_PACKET_TYPE/APP_NAME', args.time_feat_micro, 'RPL_RANK', 'RPL_VERSION']]
        ranks_vers = ranks_vers.rename(columns={args.time_feat_micro: args.time_feat_sec})

        name = args.out_feat_files + '/' + args.scenario + '/simulation-' + \
               filenames[file_index].split("/")[-1].split("\\")[-1].split(".")[0] + \
               '/' + 'RANKS_VERS.csv'
        # Create folder of simulation under the corresponding scenario in the folder containing features
        if not os.path.exists('/'.join(name.split("/")[:-1])):
            os.makedirs('/'.join(name.split("/")[:-1]))
        ranks_vers.to_csv(name, index=False, header=True)

    toc = time.time()
    # print('Time taken with parallelization: {}'.format(toc-tic))


if __name__ == '__main__':
    # main()
    parallelized_main()
