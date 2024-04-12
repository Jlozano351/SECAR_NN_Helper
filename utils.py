# Functions to create/modify/run cosy files

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pickle 
#################################################################



#NEURAL NETWORK MODELS
#__________________________________________________________________________________________________________________________________________

#Body of the NN with 6 layers, 50 neurons each WITH SIN RANGE DETERMINER FUNCTION
class sin_NN_model_3D(nn.Module):
    def __init__(self):
        super(sin_NN_model_3D, self).__init__()
        self.fc1 = nn.Linear(8, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        # Apply sine transformation to the first four output neurons
        x_transformed = x.clone()
        x_transformed[:, 0] = torch.sin(x[:, 0]*50) * 0.01 #<-- X
        x_transformed[:, 1] = torch.sin(x[:, 1]*50) * 0.01 #<-- aX
        #x_transformed[:, 1] = 0
        x_transformed[:, 2] = torch.sin(x[:, 2]*50) * 0.01 #<-- Y
        x_transformed[:,3:] = 0
        #x_transformed[:, 3] = torch.sin(x[:, 3]*50) * 0.01 #<-- aY
        #x_transformed[:, 4] = 0
        #x_transformed[:, 5] = torch.sin(x[:, 3]*50) * 0.003 #<-- dE
        #x_transformed[:,6:] = 0

        return x_transformed
    
#Body of the NN with 6 layers, 50 neurons each WITH SIGMOID RANGE DETERMINER FUNCTION 
class sigmoid_NN_model_3D(nn.Module):
    def __init__(self):
        super(sigmoid_NN_model_3D, self).__init__()
        self.fc1 = nn.Linear(8, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        # Apply sine transformation to the first four output neurons
        x_transformed = x.clone()
        x_transformed[:, 0] = (torch.sigmoid(x[:, 0])-0.5) * 0.02 #<-- X 
        x_transformed[:, 1] = (torch.sigmoid(x[:, 1])-0.5) * 0.02 #<-- aX
        #x_transformed[:, 1] = 0
        x_transformed[:, 2] = (torch.sigmoid(x[:, 2])-0.5) * 0.02 #<-- Y
        x_transformed[:,3:] = 0
        #x_transformed[:, 3] = torch.sin(x[:, 3]*50) * 0.01 #<-- aY
        #x_transformed[:, 4] = 0
        #x_transformed[:, 5] = torch.sin(x[:, 3]*50) * 0.003 #<-- dE
        #x_transformed[:,6:] = 0

        return x_transformed
    
#Body of the NN with 6 layers, 50 neurons each WITH SIGMOID RANGE DETERMINER FUNCTION     
class sigmoid_NN_model_4D(nn.Module):
    def __init__(self):
        super(sigmoid_NN_model_4D, self).__init__()
        self.fc1 = nn.Linear(8, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        # Apply sine transformation to the first four output neurons
        x_transformed = x.clone()
        x_transformed[:, 0] = (torch.sigmoid(x[:, 0])-0.5) * 0.02 #<-- X 
        x_transformed[:, 1] = (torch.sigmoid(x[:, 1])-0.5) * 0.02 #<-- aX
        x_transformed[:, 2] = (torch.sigmoid(x[:, 2])-0.5) * 0.02 #<-- Y
        x_transformed[:, 3] = (torch.sigmoid(x[:, 3])-0.5) * 0.02 #<-- Y
        x_transformed[:,4:] = 0

        return x_transformed
    

#Body of the NN with 6 layers, 50 neurons each WITH SIGMOID RANGE DETERMINER FUNCTION     
class sigmoid_NN_model_5D(nn.Module):
    def __init__(self):
        super(sigmoid_NN_model_5D, self).__init__()
        self.fc1 = nn.Linear(8, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        # Apply sigmoid transformation to the first four output neurons
        x_transformed = x.clone()
        x_transformed[:, 0] = (torch.sigmoid(x[:, 0])-0.5) * 0.02 #<-- X 
        x_transformed[:, 1] = (torch.sigmoid(x[:, 1])-0.5) * 0.02 #<-- aX
        x_transformed[:, 2] = (torch.sigmoid(x[:, 2])-0.5) * 0.02 #<-- Y
        x_transformed[:, 3] = (torch.sigmoid(x[:, 3])-0.5) * 0.02 #<-- Y
        x_transformed[:, 4] = 0
        x_transformed[:, 5] = (torch.sigmoid(x[:, 5])) * 0.035 #<-- dE 
        x_transformed[:,6:] = 0

        return x_transformed
    
#Body of the NN with 6 layers, 50 neurons each WITH SIGMOID RANGE DETERMINER FUNCTION     
#THE MAIN CHANGE OF THIS MODEL IS THAT IT CAN TAKE NEGATIVE VALUES OF dE
class sigmoid_NN_model_5D_2(nn.Module):
    def __init__(self):
        super(sigmoid_NN_model_5D_2, self).__init__()
        self.fc1 = nn.Linear(8, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        # Apply sigmoid transformation to the first four output neurons
        x_transformed = x.clone()
        x_transformed[:, 0] = (torch.sigmoid(x[:, 0])-0.5) * 0.02 #<-- X 
        x_transformed[:, 1] = (torch.sigmoid(x[:, 1])-0.5) * 0.02 #<-- aX
        x_transformed[:, 2] = (torch.sigmoid(x[:, 2])-0.5) * 0.02 #<-- Y
        x_transformed[:, 3] = (torch.sigmoid(x[:, 3])-0.5) * 0.02 #<-- aY
        x_transformed[:, 4] = 0
        x_transformed[:, 5] = (torch.sigmoid(x[:, 5])-0.5) * 0.06 #<-- dE 
        x_transformed[:,6:] = 0

        return x_transformed
    

#Body of the NN with 6 layers, 50 neurons each WITH SIGMOID RANGE DETERMINER FUNCTION     
#THE MAIN CHANGE OF THIS MODEL IS THAT IT CAN TAKE BIGGER VALUES FOR X, Y, aX, AND aY
class sigmoid_NN_model_5D_3(nn.Module):
    def __init__(self):
        super(sigmoid_NN_model_5D_3, self).__init__()
        self.fc1 = nn.Linear(8, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        # Apply sigmoid transformation to the first four output neurons
        x_transformed = x.clone()
        x_transformed[:, 0] = (torch.sigmoid(x[:, 0])-0.5) * 0.06 #<-- X 
        x_transformed[:, 1] = (torch.sigmoid(x[:, 1])-0.5) * 0.06 #<-- aX
        x_transformed[:, 2] = (torch.sigmoid(x[:, 2])-0.5) * 0.06 #<-- Y
        x_transformed[:, 3] = (torch.sigmoid(x[:, 3])-0.5) * 0.06 #<-- Y
        x_transformed[:, 4] = 0
        x_transformed[:, 5] = (torch.sigmoid(x[:, 5])-0.5) * 0.06 #<-- dE 
        x_transformed[:,6:] = 0

        return x_transformed
    

    #Body of the NN with 6 layers, 50 neurons each WITH SIGMOID RANGE DETERMINER FUNCTION     
#THE MAIN CHANGE OF THIS MODEL IS THAT IT CAN TAKE BIGGER VALUES FOR X, Y, aX, AND aY
class sigmoid_NN_model_5D_4(nn.Module):
    def __init__(self):
        super(sigmoid_NN_model_5D_4, self).__init__()
        self.fc1 = nn.Linear(8, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        # Apply sigmoid transformation to the first four output neurons
        x_transformed = x.clone()
        x_transformed[:, 0] = (torch.sigmoid(x[:, 0])-0.5) * 0.060 #<-- X 
        x_transformed[:, 1] = (torch.sigmoid(x[:, 1])-0.5) * 0.055 #<-- aX
        x_transformed[:, 2] = (torch.sigmoid(x[:, 2])-0.5) * 0.016 #<-- Y
        x_transformed[:, 3] = (torch.sigmoid(x[:, 3])-0.5) * 0.06 #<-- aY
        x_transformed[:, 4] = 0
        x_transformed[:, 5] = (torch.sigmoid(x[:, 5])-0.5) * 0.060 #<-- dE 
        x_transformed[:,6:] = 0

        return x_transformed
#__________________________________________________________________________________________________________________________________________

#__________________________________________________________________________________________________________________________________________

#MANAGEMENT OF DATA.
#__________________________________________________________________________________________________________________________________________
	
# This function compresses or expands the KDE's margins as to keep all of the data from the training data within range.
# If set is set to true it returns the standard KDE positions and bandwidth values.  
def bandloc(tensor=None, margins_in_x=0.05, margins_in_y=0.05, pixels_in_x=150, pixels_in_y=150, set_it=False):
    if tensor is None and not set_it:
        raise ValueError("If you want to change margins, please set the variable tensor to the tensor you want to find the KDE of, else set variable set_it to True.")

    if not set_it:
        b = torch.max(torch.abs(tensor)).item()
        if b <= margins_in_x:
            b = margins_in_x
    else:
        a = margins_in_x
        b = margins_in_y

    x_min, x_max, y_min, y_max = -a, a, -b, b
    X_, Y_ = np.mgrid[x_min:x_max:pixels_in_x*1j, y_min:y_max:pixels_in_y*1j] 
    positions_ = torch.tensor(np.vstack([X_.ravel(), Y_.ravel()])).T.double()

    # Ensure positions_ is a PyTorch tensor
    if not isinstance(positions_, torch.Tensor):
        positions_ = torch.tensor(positions_).double()

    return positions_    



#This function loads a CSV into a torch tensor.
def csv_to_torch_tensor(csv_file_path):
    if not isinstance(csv_file_path, str):
        raise TypeError("csv parameter must be a string")
    
    data_array = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)
    dataset = torch.tensor(data_array, dtype=torch.double)
    return dataset


# This function creates a 2xN tensor from two 1D N long tensors which is what the Neural Network takes in as input.
def loader(x, y, df=None):
    if df is not None:
        if not isinstance(x, str) or not isinstance(y, str):
            raise ValueError("x and y must be column names when df is provided")
        x_data = df[x].values
        y_data = df[y].values
        loaded = np.stack((x_data, y_data), axis=1)
    else:
        if not (torch.is_tensor(x) and torch.is_tensor(y)):
            raise ValueError("x and y must be 1D tensors when df is not provided")
        x_data = x
        y_data = y
        loaded = torch.stack((x_data, y_data), dim=1)
    
    return loaded

# Save important loss values lists to a pickle file
def loss_values_lists_to_pickle_3D_OLD(KL_storer, KL_storerA, KL_storerB, distance_storer, dataset_number):
    with open(f'PKL lists/Total_KL_list_dataset{dataset_number}.pkl', 'wb') as file:
        pickle.dump(KL_storer, file)

    with open(f'PKL lists/KL_list_matrix_A_dataset{dataset_number}.pkl', 'wb') as file:
        pickle.dump(KL_storerA, file)

    with open(f'PKL lists/KL_list_matrix_B_dataset{dataset_number}.pkl', 'wb') as file:
        pickle.dump(KL_storerB, file)

    with open(f'PKL lists/distance_list_dataset{dataset_number}.pkl', 'wb') as file:
        pickle.dump(distance_storer, file)


# Load the loss lists from a pickle file for later graphing
def load_loss_values_lists_from_pickle_3D_OLD(dataset_number):
    KL_storer = []
    KL_storerA = []
    KL_storerB = []
    distance_storer = []

    with open(f'PKL lists/Total_KL_list_dataset{dataset_number}.pkl', 'rb') as file:
        KL_storer = pickle.load(file)

    with open(f'PKL lists/KL_list_matrix_A_dataset{dataset_number}.pkl', 'rb') as file:
        KL_storerA = pickle.load(file)

    with open(f'PKL lists/KL_list_matrix_B_dataset{dataset_number}.pkl', 'rb') as file:
        KL_storerB = pickle.load(file)

    with open(f'PKL lists/distance_list_dataset{dataset_number}.pkl', 'rb') as file:
        distance_storer = pickle.load(file)

    return KL_storer, KL_storerA, KL_storerB, distance_storer


# Save important loss values lists to a pickle file
def loss_values_lists_to_pickle(KL_individual_storers, dataset_number, number_of_matrices, number_of_dimensions, custom_filename = None):
    if custom_filename is None:
        with open(f'PKL lists/KL_individual_storers_dataset{dataset_number}_{number_of_matrices}_matrices_{number_of_dimensions}D.pkl', 'wb') as file:
            pickle.dump(KL_individual_storers, file)

    else:
        with open(custom_filename, 'wb') as file:
            pickle.dump(KL_individual_storers, file)

# Load the loss lists from a pickle file for later graphing
def load_loss_values_lists_from_pickle(dataset_number, number_of_matrices, number_of_dimensions, custom_filename = None):
    KL_individual_storers = []

    if custom_filename is None:
        with open(f'PKL lists/KL_individual_storers_dataset{dataset_number}_{number_of_matrices}_matrices_{number_of_dimensions}D.pkl', 'rb') as file:
            KL_individual_storers = pickle.load(file)

    else:
        with open(custom_filename, 'rb') as file:
            KL_individual_storers = pickle.load(file)

    return KL_individual_storers


#__________________________________________________________________________________________________________________________________________
#READING AND MULTIPLYING THE COSY MATRICES
#__________________________________________________________________________________________________________________________________________
#this function is the one that reads the COSY maps for simulation of the SECAR's quadrupoles
def read_map(filename):
	"""
	The full precision cosy maps are a pain in the ass to read.
	"""

	total_coeff = []
	total_power = []
	max_len = 0

	with open(filename, "r") as f:
		coeff = []
		power = []

		for _ in range(3):
			next(f)
		for line in f:
			if "---" in line:
				if len(coeff) > max_len:
					max_len = len(coeff)
				total_coeff.append(coeff)
				total_power.append(power)
				coeff = []
				power = []
			elif "I" in line:
				pass
			else:
				line = line.split()
				temp = line[3:11]
				coeff.append(float(line[1]))
				power.append(np.asarray(temp, dtype="float64"))
	# Make this a square array, pad with zeros
	coeff_array = np.zeros((8, max_len))
	power_array = np.zeros((8, max_len, 8))

	for i in range(8):
		length = len(total_coeff[i])
		coeff_array[i, :length] = total_coeff[i]
		power_array[i, :length, :] = total_power[i]

	coeff_array_torch = torch.from_numpy(coeff_array).double()
	power_array_torch = torch.from_numpy(power_array).double()

	return coeff_array_torch, power_array_torch



# Function that performs multiplication of transport matrix by beam array
# x0 has dimensions [number of samples, beam array dimension]
def transport(x0, map_coeff, map_power):
    x = torch.zeros_like(x0, dtype=torch.double)
    n = map_coeff.shape[0]
    m = map_coeff.shape[1]

    for i in range(n):
        mask = (map_coeff[i] != 0)
        if mask.any():
            power_temp = torch.prod(x0[:, None]**map_power[i][None, mask], dim=2) * map_coeff[i][mask]
            x[:, i] += torch.sum(power_temp, dim=1, dtype=torch.double)

    return x

#This little function packs all the lists for loss to display for graphing.
def pack_lists(a, b, c):
    return [[a[i], b[i], c[i]] for i in range(len(a))]

# Performs beam transport to end given a beam array
def transportTotal(x0, i):
	coeff_array, power_array = read_map(str(i) + '/fort.50')  # DL1
	x1 = transport(x0, coeff_array, power_array)
	coeff_array, power_array = read_map(str(i) + '/fort.51')  # Q1
	x2 = transport(x1, coeff_array, power_array)
	coeff_array, power_array = read_map(str(i) + '/fort.52')  # toFP1
	x3 = transport(x2, coeff_array, power_array)
	return (x3)

#__________________________________________________________________________________________________________________________________________


#KDE GENERATION
#__________________________________________________________________________________________________________________________________________

# This is the gaussian KDE function that the pytorch paper we have been using as a source used for their model
class KDEGaussian(nn.Module):
    def __init__(self, bandwidth, locations=None):
        super(KDEGaussian, self).__init__()
        self.bandwidth = bandwidth
        self.locations = locations.float()  # Convert the input 'locations' to float data type

    def forward(self, samples, locations=None):
        samples = samples.float()  # Convert the input 'samples' to float data type
        if locations is None:
            locations = self.locations  # If 'locations' is not provided in the forward call, use the one from the initialization
        diff = samples.unsqueeze(-2) - locations.unsqueeze(-3)  # Compute the differences between 'samples' and 'locations'
        squared_diff = torch.sum(diff ** 2, dim=-1)  # Compute the squared differences along the last dimension
        out = (-squared_diff / self.bandwidth ** 2).exp().sum(dim=-2)  # Compute the exponentials and sum along the second-last dimension
        norm = out.sum(dim=-1)  # Calculate the normalization term by summing along the last dimension
        return out / norm.unsqueeze(-1)  # Return the KDE estimate divided by the normalization term
    

# This is the gaussian KDE function that the pytorch paper we have been using as a source used for their model
class KDEGaussian2(nn.Module):
    def __init__(self, bandwidth, locations=None):
        super(KDEGaussian2, self).__init__()
        self.bandwidth = bandwidth
        self.locations = locations.float()  # Convert the input 'locations' to float data type

    def forward(self, samples, locations=None):
        samples = samples.float()  # Convert the input 'samples' to float data type
        if locations is None:
            locations = self.locations  # If 'locations' is not provided in the forward call, use the one from the initialization

        #only taking into account the data within the margins
        mask = (torch.abs(samples[:, 0]) < torch.max(torch.abs(locations[:,0]))) & (torch.abs(samples[:, 1]) < torch.max(torch.abs(locations[:,1])))
        samples = samples[mask] 

        diff = samples.unsqueeze(-2) - locations.unsqueeze(-3)  # Compute the differences between 'samples' and 'locations'
        squared_diff = torch.sum(diff ** 2, dim=-1)  # Compute the squared differences along the last dimension
        out = (-squared_diff / self.bandwidth ** 2).exp().sum(dim=-2)  # Compute the exponentials and sum along the second-last dimension
        norm = out.sum(dim=-1)  # Calculate the normalization term by summing along the last dimension
        return out / norm.unsqueeze(-1)  # Return the KDE estimate divided by the normalization term
#__________________________________________________________________________________________________________________________________________
  

#THIS PART IS CREATION OF BEAMS, IT IS NOT USED IN THE MODEL, HOWEVER IT IS USEFUL IF I NEED TO EXPLAIN HOW I CREATED THE BEAMS.
#__________________________________________________________________________________________________________________________________________
#Creates a new input_to_nn if neeeded.
def input_to_nn_creator(n):
    input_to_nn = torch.normal(mean=0, std=1, size=(n, 8)).double()
    return input_to_nn

#This function creates canonical_beam for a 3D (X, Y, aX)
def canonical_beam_3D_creator(dataset):
    data = pd.read_csv(f"./initial settings/data_needed_3D-4D.csv")
    data.columns = ["x1", "x2", "y1_1", "y2_1", "y1_2", "y2_2", "y1_3", "y2_3", "aX"]
    canonical_beam = np.zeros((200, 8))
    canonical_beam[:, 0] = data[f"y1_{dataset}"]
    canonical_beam[:, 1] = data["aX"]
    canonical_beam[:, 2] = data[f"y2_{dataset}"]
    canonical_beam = torch.from_numpy(canonical_beam)

    return canonical_beam


#This function creates canonical_beam for a 4D (X, Y, aX, aY)
def canonical_beam_4D_creator(dataset):
    data = pd.read_csv(f"./initial settings/data_needed_3D-4D.csv")
    data.columns = ["x1", "x2", "y1_1", "y2_1", "y1_2", "y2_2", "y1_3", "y2_3", "aX"]
    canonical_beam = np.zeros((200, 8))
    canonical_beam[:, 0] = data[f"y1_{dataset}"]
    canonical_beam[:, 1] = data["aX"]
    canonical_beam[:, 2] = data[f"y2_{dataset}"]
    canonical_beam[:, 3] = data["aX"]
    canonical_beam = torch.from_numpy(canonical_beam)

    return canonical_beam

#This function creates canonical_beam for a 4D (X, Y, aX, aY)
def canonical_beam_5D_creator(dataset):
    data = pd.read_csv(f"./initial settings/data_needed_5D.csv")
    canonical_beam = np.zeros((200, 8))
    canonical_beam[:, 0] = data[f"y1_{dataset}"]
    canonical_beam[:, 1] = data["aX"]
    canonical_beam[:, 2] = data[f"y2_{dataset}"]
    canonical_beam[:, 3] = data["aX"]
    canonical_beam[:, 5] = data["dE"]
    canonical_beam = torch.from_numpy(canonical_beam)

    return canonical_beam


#This function creates the list that contains all of the KDEs of the postCOSY_canon_beams for the 10 matrices Ruchi sent me.
def postCOSY_canon_beam_and_canonical_density_creator_matrix_A_to_J(canonical_beam, number_of_matrices, bandwidth):
    if not isinstance(number_of_matrices, int) or number_of_matrices > 10:
        raise ValueError("number_of_matrices must be an integer not exceeding 10")

    postCOSY_canon_beam_list = []
    loadedFinal_list = []

    for i in range(number_of_matrices):
        matrix_filename = f'matrix {"ABCDEFGHIJ"[i]}.txt'
        coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
        postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)
        loadedFinal1 = loader(postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2])
        # Condition to filter loadedFinal1
        condition = ((loadedFinal1[:, 0] >= -0.05000) & (loadedFinal1[:, 0] <= 0.05000) & 
                     (loadedFinal1[:, 1] >= -0.05000) & (loadedFinal1[:, 1] <= 0.05000))
        loadedFinal = loadedFinal1[condition]
        postCOSY_canon_beam_list.append(postCOSY_canon_beam)
        loadedFinal_list.append(loadedFinal)

    positions_ = bandloc(set_it=True) #<----- the preestablished values of bandwidth (0.009), and positions to evaluate KDE (-+0.05 m in X and Y) DON'T CHANGE
    kde = KDEGaussian2(bandwidth=bandwidth, locations=positions_) #<----- This is the KDE function itself.

    KDE_list = []
    for i in loadedFinal_list:
        KDE_list.append(kde(i))

    return KDE_list, postCOSY_canon_beam_list

#This function creates the list that contains all of the KDEs of the postCOSY_canon_beams to the NEW 49 matrices sent to me.
def postCOSY_canon_beam_and_canonical_density_creator_matrix_1_to_49(canonical_beam, number_of_matrices, bandwidth):
    if not isinstance(number_of_matrices, int) or number_of_matrices > 49:
        raise ValueError("number_of_matrices must be an integer not exceeding 10")

    postCOSY_canon_beam_list = []
    loadedFinal_list = []

    for i in range(number_of_matrices):
        matrix_filename = f'Matrix {i+1} 15 Feb.txt'
        coeff_array, power_array = read_map(f'./matrices2/{matrix_filename}')  
        postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)
        loadedFinal1 = loader(postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2])
        # Condition to filter loadedFinal1
        condition = ((loadedFinal1[:, 0] >= -0.05000) & (loadedFinal1[:, 0] <= 0.05000) & 
                     (loadedFinal1[:, 1] >= -0.05000) & (loadedFinal1[:, 1] <= 0.05000))
        loadedFinal = loadedFinal1[condition]
        postCOSY_canon_beam_list.append(postCOSY_canon_beam)
        loadedFinal_list.append(loadedFinal)

    positions_ = bandloc(set_it=True) #<----- the preestablished values of bandwidth (0.009), and positions to evaluate KDE (-+0.05 m in X and Y) DON'T CHANGE
    kde = KDEGaussian2(bandwidth=bandwidth, locations=positions_) #<----- This is the KDE function itself.

    KDE_list = []
    for i in loadedFinal_list:
        KDE_list.append(kde(i))

    return KDE_list, postCOSY_canon_beam_list

#This function creates the list that contains all of the KDEs of the postCOSY_canon_beams to the NEW 49 matrices sent to me.
def postCOSY_canon_beam_and_canonical_density_creator(canonical_beam, number_of_matrices, scenario_file ,bandwidth, margins_in_x=0.05, margins_in_y=0.05, pixels_in_x=150, pixels_in_y=150):
    if not isinstance(number_of_matrices, int) or number_of_matrices > 49:
        raise ValueError("number_of_matrices must be an integer not exceeding 10")

    postCOSY_canon_beam_list = []
    loadedFinal_list = []

    for i in range(number_of_matrices):
        with open(scenario_file, "r") as f:
            lines = f.readlines()
        coeff_array, power_array = read_map(lines[i][:-1])  
        postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)
        loadedFinal1 = loader(postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2])
        # Condition to filter loadedFinal1
        condition = ((loadedFinal1[:, 0] >= -0.05000) & (loadedFinal1[:, 0] <= 0.05000) & 
                     (loadedFinal1[:, 1] >= -0.05000) & (loadedFinal1[:, 1] <= 0.05000))
        loadedFinal = loadedFinal1[condition]
        postCOSY_canon_beam_list.append(postCOSY_canon_beam)
        loadedFinal_list.append(loadedFinal)

    positions_ = bandloc(margins_in_x = margins_in_x, margins_in_y = margins_in_y, pixels_in_x = pixels_in_x, pixels_in_y = pixels_in_y, set_it=True) #<----- the preestablished values of bandwidth (0.009), and positions to evaluate KDE (-+0.05 m in X and Y) DON'T CHANGE
    kde = KDEGaussian2(bandwidth=bandwidth, locations=positions_) #<----- This is the KDE function itself.

    KDE_list = []
    for i in loadedFinal_list:
        KDE_list.append(kde(i))

    return KDE_list, postCOSY_canon_beam_list

#This function creates the list that contains all of the KDEs of the postCOSY_canon_beams for the matrices Fernando suggested to train on in the day of 12/14/2023.
def postCOSY_canon_beam_and_canonical_density_creator_matrix_ABCDGHI(canonical_beam, number_of_matrices, bandwidth):
    if not isinstance(number_of_matrices, int) or number_of_matrices > 10:
        raise ValueError("number_of_matrices must be an integer not exceeding 10")

    postCOSY_canon_beam_list = []
    loadedFinal_list = []

    for i in range(number_of_matrices):
        matrix_filename = f'matrix {"ABCDGHI"[i]}.txt'
        coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
        postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)
        loadedFinal1 = loader(postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2])
        # Condition to filter loadedFinal1
        condition = ((loadedFinal1[:, 0] >= -0.05000) & (loadedFinal1[:, 0] <= 0.05000) & 
                     (loadedFinal1[:, 1] >= -0.05000) & (loadedFinal1[:, 1] <= 0.05000))
        loadedFinal = loadedFinal1[condition]
        postCOSY_canon_beam_list.append(postCOSY_canon_beam)
        loadedFinal_list.append(loadedFinal)

    positions_ = bandloc(set_it=True) #<----- the preestablished values of bandwidth (0.009), and positions to evaluate KDE (-+0.05 m in X and Y) DON'T CHANGE
    kde = KDEGaussian(bandwidth=bandwidth, locations=positions_) #<----- This is the KDE function itself.

    KDE_list = []
    for i in loadedFinal_list:
        KDE_list.append(kde(i))

    return KDE_list, postCOSY_canon_beam_list

#__________________________________________________________________________________________________________________________________________

#CALCULATION OF DISTANCE
#__________________________________________________________________________________________________________________________________________

#This function calculates the distance value of the loss function.
def distance_calculator(postCOSY_pred_beam, postCOSY_canon_beam):
    #This portion of the code finds all of the points of postCOSY_pred_beam that are out of the established +-0.05 viewer margins
    out_of_bounds_on_x = (postCOSY_pred_beam[:, 0] >= 0.05) | (postCOSY_pred_beam[:, 0] <= -0.05)
    out_of_bounds_on_y = (postCOSY_pred_beam[:, 2] >= 0.05) | (postCOSY_pred_beam[:, 2] <= -0.05)
    out_of_bounds = (out_of_bounds_on_x | out_of_bounds_on_y)
    postCOSY_pred_beam_OUT_OF_RANGE = postCOSY_pred_beam[out_of_bounds]
    #This part finds the medians of the postCOSY_canon_beams that are which we are trying to make postCOSY_pred_beam look like for using in the formula for distance.
    # FORMULA FOR DISTANCE = Σ[(Xi - Vx)**2 + (Yi - Vi)**2]
    #Xi = position for points out of bounds in x
    #Vx = median of the points we are using as training data in x
    #Xi = position for points out of bounds in y
    #Vx = median of the points we are using as training data in y
    postCOSY_canon_beam_x_median = torch.median(postCOSY_canon_beam[:, 0])
    postCOSY_canon_beam_y_median = torch.median(postCOSY_canon_beam[:, 2])
    distance = torch.sum((postCOSY_pred_beam_OUT_OF_RANGE[:,0] - postCOSY_canon_beam_x_median)**2 + (postCOSY_pred_beam_OUT_OF_RANGE[:,2] - postCOSY_canon_beam_y_median)**2)
    return distance
#__________________________________________________________________________________________________________________________________________


#ALL OF THE PLOTTING FUNCTIONS ARE IN ANOTHER FILE BECAUSE THEY ARE SO LONG THAT IF I WERE TO PUT THEM HERE THE
#LIBRARY WOULD BE HARD TO READ.
#__________________________________________________________________________________________________________________________________________
#-------------------------------------------------------------------------------------------------------------------------------------------
# This functions are but legacy code from when the code was ran in numpy and not pytorch, I doubt it will be useful 
# but it cannot hurt to keep it around.

# Function that performs multiplication of transport matrix by beam array
# x0 has dimensions [number of samples, beam array dimension]
# **IN NUMPY**
def transport_numpy(x0, map_coeff, map_power):
	x = np.zeros(x0.shape, dtype="float64", order="F")
	samples = x0.shape[0]
	n = map_coeff.shape[0]  # Beam array dimensions
	m = map_coeff.shape[1]  # Number of coefficients

	for l in range(samples):  # Loop over the samples
		for i in range(n):  # Loop over beam array dimensions
			j = 0
			while j < m and map_coeff[i][j] != 0:  # Loop over coefficients
				power_temp = 1
				for k in range(n):    # Loop over beam array dimensions
					power_temp = power_temp*x0[l][k]**map_power[i][j][k]
				x[l][i] = x[l][i] + map_coeff[i][j]*power_temp
				j = j + 1
	return x



# Performs beam transport to end given a beam array
# **IN NUMPY**
def transportTotal_numpy(x0, i, coeff):
	coeff_array, power_array = read_map(str(i) + '/fort.50')  # DL1
	x1 = transport(x0, coeff_array, power_array)
	coeff_array, power_array = read_map(str(i) + '/fort.51')  # Q1
	for c in coeff:  # Modifying transport matrix element by element
		coeff_array[c[0]][c[1]] = c[2]
	x2 = transport(x1, coeff_array, power_array)
	coeff_array, power_array = read_map(str(i) + '/fort.52')  # toFP1
	x3 = transport(x2, coeff_array, power_array)
	return (x3)

#-------------------------------------------------------------------------------------------------------------------------------------------