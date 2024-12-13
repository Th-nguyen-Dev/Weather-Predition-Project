def import_dataframes_dict(folder_path):
    import os
    import pandas as pd
    dataframes_dict = {}
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df_name = file.split('.')[0]
            dataframes_dict[df_name] = pd.read_csv(folder_path + file)
    return dataframes_dict

def create_node_feature_tensor(dataframes_dict, offset=0):
    import torch
    import numpy as np
    
    if dataframes_dict == {}:
        return torch.tensor([])
    
    array_3D = []
    for df in dataframes_dict.values():
        df_features = df.to_numpy()
        array_3D.append(df_features) 
        
    #shape [n_nodes, n_rows, n_features]
    base_tensor = torch.tensor(np.array(array_3D))
    
    #shape [n_rows, n_nodes, n_features]
    permute_tensor = base_tensor.permute(1, 0, 2)
    
    # If there is offset, remove the first offset rows
    if offset > 0:
        permute_tensor = permute_tensor[offset:]
    
    return permute_tensor


def create_edges_tensor_2D(dataframes_dict):
    import torch
    import numpy as np
    keys_length = len(list(dataframes_dict.keys()))
    
    edges_start = []
    edges_end = []
    
    # Create all possible edges
    for i in range(keys_length):
        for j in range(i, keys_length):
            if i != j:
                edges_start.append(i)
                edges_end.append(j)
                edges_start.append(j)
                edges_end.append(i)
                
    edges_array = np.array([edges_start, edges_end])
    edges_tensor = torch.tensor(edges_array)
    return edges_tensor
