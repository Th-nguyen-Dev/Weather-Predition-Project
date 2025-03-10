# Weather Prediction Project

## Authors
The Hung Nguyen & Ritesh Samal

## Overview
This project implements a comprehensive weather prediction system using deep learning techniques inspired by Google's GraphCast AI. The system processes and analyzes weather data spanning from January 1, 2020 to October 27, 2024 (1,732 days) from multiple weather stations, and builds models that capture both temporal and spatial dynamics to predict future weather conditions.

## Project Structure
- `data_process_stage_*.ipynb`: Sequential data processing pipeline
- `model/model_refactor_version_2.ipynb`: Main model implementation
- `model/model_refactor_version_Test_Purpose.ipynb`: Model testing variants
- `outdated/`: Previous model implementations
- `CODE_DOCUMENTATION.md`: Detailed project documentation
- `processed-final-data/`: Processed weather station data
- `draft-final-data/`: Intermediate processed data
- `getloc.sh`: Script to retrieve location data

## Data Processing Pipeline
The data processing involves five sequential stages:

1. **Stage 1: Data Consolidation**
   - Ingests raw weather data and organizes by station
   - Processes data chronologically to maintain temporal relationships
   - Outputs to `process-by-station/` directory

2. **Stage 2: Feature Processing**
   - Cleans and standardizes key weather parameters
   - Handles temperature measurements, pressure readings, precipitation, etc.
   - Filters stations based on data quality (reduced from 42 to 27 stations)
   - Outputs to `draft-final-data/` directory

3. **Stage 3: Feature Engineering**
   - Enhances dataset with engineered features
   - Processes categorical weather types
   - Implements cyclical encoding for temporal features
   - Outputs to `processed-data/` directory

4. **Stage 4: Geographic Integration**
   - Incorporates geographical coordinates
   - Creates distance-based spatial relationships between stations
   - Enhances the dataset with location metadata

5. **Stage 5: Final Processing**
   - Ensures complete geographical coverage for all 27 stations
   - Prepares data for model input
   - Outputs to `processed-final-data/` directory

## Neural Network Models
The project implements several neural network architectures:

1. **Transformer Module**
   - Captures temporal relationships in the data
   - Uses multi-head attention mechanisms

2. **Graph Neural Network (GNN) Module**
   - Captures spatial relationships between weather stations
   - Uses geographical distances to create graph edges

3. **Hybrid Models**
   - `model_parallel`: Parallel processing of temporal and spatial features
   - `model_transformer_gnn`: Sequential processing with transformer followed by GNN
   - `model_gnn_transformer`: Sequential processing with GNN followed by transformer
   - `model_single_transformer`: Temporal-only model
   - `model_single_gnn`: Spatial-only model

## Model Features
- Hyperparameter tuning with scheduler
- Learning rate adjustment
- Model evaluation using MSE and MAE metrics
- Training visualization
- Model saving and loading functionality

## Dataset Characteristics
- 27 high-quality weather stations
- 69 features including:
  - Geographical features (latitude, longitude, elevation)
  - Meteorological measurements (temperature, pressure, wind, etc.)
  - Temporal features (cyclical day of year)
  - Categorical features (weather types, sky conditions)
  - Derived features (relative humidity, pressure tendencies)

## Future Work
As outlined in `idea_for_improvement.txt`, future work will explore a 3D dimensional approach to neural networks where nodes connect on both spatial and temporal layers. This means that nodes in the past can connect to nodes in the present with their own custom weights, increasing the dimensionality of the graph from spatial dimensions to a time-space dimension. The model will be updated with more hidden channels and more transformer layers by training on the cloud. The data processing and data splitting will be refactored and unit tested to ensure correct output. The neural network architecture will be redesigned and optimized to ensure that the graph neural network can process time as well. Additionally, the model will be tested with different batch sizes to observe performance differences, and efforts will be made to configure the model to predict the nth next day. Finally, a research paper will be written with an improved, more standardized dataset, extensive hyperparameter tuning, and higher processing capabilities.

## Requirements
- Python with PyTorch, PyTorch Geometric
- CUDA support for GPU acceleration
- Pandas, NumPy, Matplotlib
- Scikit-learn for model evaluation

## Getting Started
1. Clone the repository
2. Process data using the sequential notebooks `data_process_stage_*.ipynb`
3. Train models using `model/model_refactor_version_2.ipynb`
4. Evaluate model performance with visualizations

For detailed technical information, refer to the comprehensive `CODE_DOCUMENTATION.md` file.
