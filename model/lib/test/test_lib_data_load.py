import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import pandas as pd

from model.lib.lib_data_load import import_dataframes_dict
from model.lib.lib_data_load import create_node_feature_tensor
from model.lib.lib_data_load import create_edges_tensor_2D
import torch

test_dataframes = {
    'node_1': pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [4, 5, 6]
    }),
    'node_2': pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [4, 5, 6]
    }),
    'node_3': pd.DataFrame({
    'feature_1': [1, 2, 3],
    'feature_2': [4, 5, 6]
    }),
    'node_4': pd.DataFrame({
    'feature_1': [1, 2, 3],
    'feature_2': [4, 5, 6]
    })
}

test_empty_dataframes = {}

test_input_tensor = torch.tensor([
    [[1,4],[1,4],[1,4],[1,4]],
    [[2,5],[2,5],[2,5],[2,5]],
    [[3,6],[3,6],[3,6],[3,6]]
])

test_output_tensor_offset_1 = torch.tensor([
    [[2,5],[2,5],[2,5],[2,5]],
    [[3,6],[3,6],[3,6],[3,6]],

])

test_output_tensor_offset_2 = torch.tensor([
    [[3,6],[3,6],[3,6],[3,6]],
])

test_output_edges_tensor = torch.tensor([
    [0, 1 , 0 , 2 , 0 , 3, 1, 2, 1, 3, 2, 3],
    [1, 0 , 2, 0 , 3 , 0, 2, 1, 3, 1, 3, 2],
])
    
test_empty_tensor = torch.tensor([])
class TestDataLoad(unittest.TestCase):
    def test_create_node_feature_tensor(self):
        self.assertTrue(torch.equal(create_node_feature_tensor(test_dataframes, 0), test_input_tensor))
        
    def test_create_node_feature_tensor_offset_1(self):
        self.assertTrue(torch.equal(create_node_feature_tensor(test_dataframes, 1), test_output_tensor_offset_1))

    def test_create_node_feature_tensor_offset_2(self):
        self.assertTrue(torch.equal(create_node_feature_tensor(test_dataframes, 2), test_output_tensor_offset_2))
        
    def test_create_node_feature_tensor_empty(self):
        self.assertTrue(torch.equal(create_node_feature_tensor(test_empty_dataframes, 3), test_empty_tensor))
        
    def test_create_edges_tensor_2D(self):
        self.assertTrue(torch.equal(create_edges_tensor_2D(test_dataframes), test_output_edges_tensor))
    
    def test_create_edges_tensor_2D_empty(self):
        self.assertTrue(torch.equal(create_edges_tensor_2D(test_empty_dataframes), test_empty_tensor))    
if __name__ == '__main__':
    unittest.main()