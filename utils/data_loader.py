import pickle
import pandas as pd
import numpy as np
from os.path import join as pjoin
import datetime 

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_dataset(dataset_root, dataset_name):
    FILE_PATH = pjoin(dataset_root, dataset_name)
    if dataset_name == 'metr-la':
        FILE_SENSOR_IDS = pjoin(FILE_PATH, 'graph_sensor_ids.txt')
        FILE_SENSOR_LOC = pjoin(FILE_PATH, 'graph_sensor_locations_corrected.csv')
        FILE_ADJ_MX = pjoin(FILE_PATH, 'adj_mx.pkl')
        file_data = pjoin(FILE_PATH, 'metr-la.h5')
        
        sensor_df = pd.read_csv(FILE_SENSOR_LOC, index_col=0)
        sensor_df.columns = ['sid', 'lat', 'lng']
        _sensor_ids, _sensor_id_to_ind, adj_mx = load_graph_data(FILE_ADJ_MX)
        
    elif dataset_name == 'pems-bay':    
        FILE_PATH = '../dataset/pems-bay/'
        FILE_SENSOR_IDS = pjoin(FILE_PATH, 'graph_sensor_ids_bay.txt')
        FILE_SENSOR_LOC = pjoin(FILE_PATH, 'graph_sensor_locations_bay.csv')
        FILE_ADJ_MX = pjoin(FILE_PATH, 'adj_mx_bay.pkl')
        file_data = pjoin(FILE_PATH, 'pems-bay.h5')

        sensor_df = pd.read_csv(FILE_SENSOR_LOC, names=['sid', 'lat', 'lng'])
        _sensor_ids, _sensor_id_to_ind, adj_mx = load_graph_data(FILE_ADJ_MX)

    elif dataset_name == 'pemsd7':
        FILE_SENSOR_LOC = pjoin(FILE_PATH, 'PeMSD7_M_Station_Info.csv')
        sensor_df = pd.read_csv(FILE_SENSOR_LOC, index_col=0)
        sensor_df.columns = ['sid', 'fwy', 'dir', 'district', 'lat', 'lng']
        
        if True or (not os.path.isfile(pjoin(FILE_PATH, 'adj_mx.pkl')) or not os.path.isfile(pjoin(FILE_PATH, 'pemsd7.h5'))):
            FILE_DIST_CSV = pjoin(FILE_PATH, 'PeMSD7_W_228.csv')
            FILE_DATA_CSV = pjoin(FILE_PATH, 'PeMSD7_V_228.csv')

            _sensor_ids = sensor_df['sid'].astype(str).tolist()
            _sensor_id_to_ind = {k:i for i, k in enumerate(_sensor_ids)}
            _dist_df = pd.read_csv(FILE_DIST_CSV, header=None)
            _dist_mx = _dist_df.values /1609.34
            _sigma = 10**.5
            _adj_mx =  np.exp(- (_dist_mx / _sigma)**2)
            _adj_mx[_adj_mx < .1] = 0

            adj_mx = _adj_mx
            with open(pjoin(FILE_PATH, 'adj_mx.pkl'), 'wb') as f:
                pickle.dump([_sensor_ids, _sensor_id_to_ind, adj_mx], f, protocol=2)
            
            _data_df = pd.read_csv(FILE_DATA_CSV, header=None)
            _data_df.columns = _sensor_ids
            start_time = datetime.datetime(2012, 5, 1, 0, 0, 0)
            end_time = datetime.datetime(2012, 7, 1, 0, 0, 0)

            five_mins = datetime.timedelta(minutes=5)
            timeslot = []
            curr_time = start_time
            while curr_time < end_time:
                if curr_time.weekday() < 5:
                    timeslot.append(curr_time)
                curr_time = curr_time + five_mins

            _data_df.index = timeslot
            _data_df.to_hdf('pemsd7/pemsd7.h5', key='df')
        
        FILE_ADJ_MX = pjoin(FILE_PATH, 'adj_mx.pkl')
        file_data = pjoin(FILE_PATH, 'pemsd7.h5')
        _sensor_ids, _sensor_id_to_ind, adj_mx = load_graph_data(FILE_ADJ_MX)
    data_df = pd.read_hdf(file_data)
    sensor_ids = data_df.columns.values.astype(str)
    assert np.sum(sensor_ids == _sensor_ids) == len(sensor_ids)

    return data_df, sensor_ids, sensor_df, _sensor_id_to_ind, adj_mx