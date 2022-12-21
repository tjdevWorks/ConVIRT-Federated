
import random
from typing import Dict

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch


def process_traindata(train_path: str, policy: str):
    train_df = pd.read_csv(train_path)
    print(f'The shape of the training dataset is : {train_df.shape}')
    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
                    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    train_df = train_df.fillna(0)
    if policy == 'zero': 
        for class_name in class_names:
            train_df[class_name] = train_df[class_name].replace([1.0],1).replace([0.0, -1.0], 0).astype({class_name: int})
    elif policy == 'one':
        for class_name in class_names:
            train_df[class_name] = train_df[class_name].replace([1.0, -1.0],1).replace([0.0], 0).astype({class_name: int})
    train_df['index'] = range(1, len(train_df) + 1)
    return train_df

def reverse_dict(label_client_dict):
    my_inverted_dict = dict()
    for key, value in label_client_dict.items():
        for elem in value:
            my_inverted_dict.setdefault(elem, list()).append(key)
    return my_inverted_dict  

#Volume split - doesn't look at y at all!
#test case for partition_volume
#result = partition_volume(data, 'dirchlet', 10, 1 )
def partition_volume(data, mode: str='uniform', num_clients: int=10, scale: int=1, sample_percent=1):
    data_indices = np.arange(len(data))
    if mode == 'uniform':
        partitions = np.array_split(data_indices, num_clients)
    elif mode == 'dirchlet':    
        vol_ratios = np.random.dirichlet(np.ones(num_clients)*scale,size=1)[0]
        split_indexs = torch.round(torch.tensor(np.cumsum(vol_ratios)*len(data_indices))).type(torch.IntTensor)
        partitions = np.split(data_indices, indices_or_sections=split_indexs) 
    sampled_partitions = list(map(lambda x: np.random.choice(x, size=int(np.round(len(x)*sample_percent)), replace=False), partitions))
    results = dict(zip(np.arange(num_clients), sampled_partitions))
    return results

def partition_feature(data, feature, num_partitions, mode='nonIID'):
    #sort the data according to age
    sorted_data = data.sort_values(feature)
    #split it like volume into equal no. of sets (would be based on percentiles)
    split = np.array_split(sorted_data.index.tolist(), num_partitions)
    if mode == 'uniform':
        #redistribute them equally across all the clients
        split_of_splits = list(map(lambda x: np.array_split(x, num_partitions), split))
        print(split_of_splits)
        uniform_splits = [[] * num_partitions] * num_partitions
        for i in range(num_partitions):
            #take every age distribution
            for j in range(num_partitions):
                #take every client partition and append a part of it
                uniform_splits[j] = np.append(uniform_splits[j], split_of_splits[i][j])
        split = list(map(lambda x: x.astype(int), uniform_splits))                                          
    result = dict(zip(np.arange(0, num_partitions), split))
    return result     

#test case for partition_class
# result = partition_class(data, {} ,'single_client_per_class', 3, exclusive=False, equal_num_samples=True)     
def partition_class(data, modeparams: Dict, mode: str='single_client_per_class', num_clients: int=1, exclusive: bool=False, equal_num_samples: bool=False, min_client_samples: int=0, sample_percent=1 ):
    #distribute classes accordingly. 
    labels = modeparams['labels']
    num_labels = len(labels)
    label_indices = np.arange(num_labels)
    results = {}
    if mode == 'single_client_per_class':
        #each class will randomly go to one client and all class elements will reside on the client
        client_label_dict = {}
        for i in range(num_labels):
            client_label_dict.setdefault(i%num_clients, list()).append(i)
        print(client_label_dict)
             
        for j in range(num_clients):
            client_labels = client_label_dict[j]
            non_client_labels = np.setdiff1d(label_indices, client_labels).astype(int)
            filtered_data = data
            # for k in client_labels:
            filtered_data = filtered_data[filtered_data.loc[:, list(map(lambda x: labels[x], client_labels))].any(axis=1)]
            #Filter based on all other labels if exclusive
            if exclusive==True:   
                non_client_labels = np.setdiff1d(label_indices, client_labels).astype(int)
                # for k in non_client_labels:
                #it's going to be !( or of all other classes )
                filtered_data = filtered_data[~filtered_data.loc[:, list(map(lambda x: labels[x], non_client_labels.tolist()))].any(axis=1)]
                #filtered_data = filtered_data.loc[filtered_data[labels[k]]==0]   
            results[j] = filtered_data['index'].tolist()      

        min_num_samples = min(list(map(lambda x: len(x), results.values())))
        if min_num_samples < min_client_samples:
            raise Exception("The minimum of the client samples generated by the strategy is less than the minimum required samples")
        if equal_num_samples==True:
            generate_samples = list(map(lambda x: random.sample(x, min_num_samples), results.values()))
            results = dict(zip(np.arange(num_clients), generate_samples))
    
    results_splits = results.values()    
    sampled_results_splits = list(map(lambda x: random.sample(x, int(np.round(len(x)*sample_percent))), results_splits))
    results = dict(zip(np.arange(num_clients), sampled_results_splits))
    return results
            


