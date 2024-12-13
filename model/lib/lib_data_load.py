def import_dataframes_dict(folder_path):
    import os
    import pandas as pd
    dataframes_dict = {}
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df_name = file.split('.')[0]
            dataframes_dict[df_name] = pd.read_csv(folder_path + file)
    return dataframes_dict

def create_node_feature_tensor(dataframes_dict):
    import torch
    import numpy as np
    return_tensor = []
    for df in dataframes_dict.values():
        df_features = df.to_numpy()
        #shape [n_nodes, n_rows, n_features]  
        return_tensor.append(df_features) 
    return_tensor = torch.tensor(np.array(return_tensor))
    #shape [n_rows, n_nodes, n_features]
    return_tensor = return_tensor.permute(1, 0, 2)
    print(return_tensor.shape)
    return return_tensor