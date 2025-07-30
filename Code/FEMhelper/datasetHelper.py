import os
import numpy as np
from FEMhelper.femHelper import *
import math

resolutionConfigs = [
    {"savePath": "datasetsFNO/32_32", "size": 32},
    {"savePath": "datasetsFNO/64_64", "size": 64},
    {"savePath": "datasetsFNO/128_128", "size": 128},
    {"savePath": "datasetsFNO/256_256", "size": 256}
]


# Create a sample of the input and output data
def create_sample(C_in=3, C_out=6, problem = None, **material_params):
    materials, u, stress, strain = solve_problem(problem, **material_params)


    input_data_sets = []
    output_data_sets = []
    for config in resolutionConfigs[1:2]:
        resolution_path = config["savePath"]
        N_x = config["size"]
        N_y = config["size"]

        x_coords = np.linspace(0, 1, N_x)
        y_coords = np.linspace(0, 1, N_y)
        X, Y = np.meshgrid(x_coords, y_coords)

        input = np.zeros((N_x, N_y, C_in))
        output = np.zeros((N_x, N_y, C_out))
        
        for i in range(N_x):
            for j in range(N_y):
                input[i, j, 0] = materials(X[i, j], Y[i, j])
                output[i, j, :2] = u(X[i, j], Y[i, j])
                output[i, j, 2:] = stress(X[i, j], Y[i, j])
                if(C_in >=1):
                    input[i, j, 1] = X[i, j]
                    input[i, j, 2] = Y[i, j]       

        input_data = input.reshape((1, N_x, N_y, C_in))
        output_data = output.reshape((1, N_x, N_y, C_out))
        input_data_sets.append(input_data)
        output_data_sets.append(output_data)

    return input_data_sets, output_data_sets

# Save the input and output data to .npy files
def create_dataset(save_path, num_samples=2000, problem = None, **material_params):

    for i in range(num_samples):        
        input_data_sets, output_data_sets = create_sample(problem=problem, **material_params)

        for res in range(len(input_data_sets)):
            input_data = input_data_sets[res]
            output_data = output_data_sets[res]
            N_x = input_data.shape[1]
            save_path_res = os.path.join(save_path, f"{N_x}_{N_x}")
            input_file = os.path.join(save_path_res, 'input_data.npy')
            output_file = os.path.join(save_path_res, 'output_data.npy')

            if os.path.exists(input_file) and os.path.exists(output_file):
                # Load existing data from the file
                existing_input_data = np.load(input_file)
                existing_output_data = np.load(output_file)

                # Append the new data
                updated_input_data = np.concatenate((existing_input_data, input_data), axis=0)
                updated_output_data = np.concatenate((existing_output_data, output_data), axis=0)
            else:
                # If file does not exist, initialize with new data
                updated_input_data = input_data
                updated_output_data = output_data

            # Save the updated data to .npy files
            os.makedirs(save_path_res, exist_ok=True)
            np.save(input_file, updated_input_data)
            np.save(output_file, updated_output_data)

# Save the input and output data to .npy files 
# Note: Not used in the current implementation

def save_sample(E, u, input_file='input_data.npy', output_file= 'output_data.npy'):
    N_x = 32
    N_y = 32
    x_coords = np.linspace(0, 1, N_x)
    y_coords = np.linspace(0, 1, N_y)
    X, Y = np.meshgrid(x_coords, y_coords)
    # X = X.flatten()
    # Y = Y.flatten()
    input = np.zeros((N_x, N_y))
    output = np.zeros((N_x, N_y, 2))
    for i in range(N_x):
        for j in range(N_y):
            input[i, j] = E(X[i, j], Y[i, j])
            output[i, j] = u(X[i, j], Y[i, j])

    input_data = input.reshape((1, N_x, N_y, 1))
    output_data = output.reshape((1, N_x, N_y, 2))
    
    
    try:
        # Load existing data from the file
        existing_input_data = np.load(input_file)
        existing_output_data = np.load(output_file)

        # Append the new data
        updated_input_data = np.concatenate((existing_input_data, input_data), axis=0)
        updated_output_data = np.concatenate((existing_output_data, output_data), axis=0)
    except FileNotFoundError:
        # If file does not exist, initialize with new data
        updated_input_data = input_data
        updated_output_data = output_data
    
    # # Save the input and output data to .npy files
    np.save(input_file, updated_input_data)
    np.save(output_file, updated_output_data)
