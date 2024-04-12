import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn

from .utils import (bandloc, read_map, loader, transport, KDEGaussian)



# This function plots a given KDE when using the ax method
def KDE_plotter_ax(axes, tensor, a=150, b=150, title=None):
    hi = torch.squeeze(tensor.reshape(a, b).T.detach())
    hi = torch.flip(hi, [0])
    sns.heatmap(hi, cmap='coolwarm', ax=axes)
    if title is not None:
        axes.set_title(title)


# This function plots a given KDE
def KDE_plotter(tensor, a = 150, b = 150, title = None):
	# We reshape the data to fit the expected image size, transpose it to 
    hi = torch.squeeze(tensor.reshape(a, b).T.detach())
    # Mirror the image along the y-axis because for some reason the output is originally mirrored in said axis
    hi = torch.flip(hi, [0])
    sns.heatmap(hi, cmap='coolwarm')
    if title is not None:
        plt.title(title)

#I created this function for rapid evaluation of a 3D canonical_beam when needed
def canonical_beam_3D_evaluator(canonical_beam):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
    """
    b = torch.max(torch.abs(canonical_beam))

    # Create a 2x2 plot matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot x vs y - canonical_beam
    axes[0].scatter(canonical_beam[:,0], canonical_beam[:,2])
    axes[0].set_title("x vs y - canonical_beam")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    
    # Plot x vs aX - canonical_beam
    axes[1].scatter(canonical_beam[:,0], canonical_beam[:,1])  # Replace 'y' with 'aX' or provide the actual data for 'aX'
    axes[1].set_title("x vs aX - canonical_beam")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("aX")

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

#This function displays the datasets
def plotting_datasets(csv_file_path):
    def margin_setter(datax, datay):
        # Determines the margin for a scatter plot based on the maximum absolute values of input data.
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        return np.max([a, b])

    def scatterplot_scatter(ax, x1, x2, label1, x_label, y_label):
        # Plots a scatter plot on the specified axis.
        ax.scatter(x1, x2, label=label1, s=6)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        margin = margin_setter(x1, x2) * 1.2
        ax.set_xlim(-margin, margin)

    # Load data from CSV file
    data_array = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)
    dataset = torch.tensor(data_array, dtype=torch.double)

    # Create subplots for a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))

    # Extract the dataset name from the CSV file name
    beam1_name = csv_file_path[:-4] 

    # Set the main title for the entire figure
    plt.suptitle(beam1_name, fontsize=30)

    # Plot scatter plots for different combinations of columns
    scatterplot_scatter(axes[0, 0], dataset[:, 0], dataset[:, 2], beam1_name, "x", "y")
    scatterplot_scatter(axes[0, 1], dataset[:, 0], dataset[:, 1], beam1_name, "x", "aX")
    scatterplot_scatter(axes[1, 0], dataset[:, 2], dataset[:, 3], beam1_name, "y", "aY")
    scatterplot_scatter(axes[1, 1], dataset[:, 0], dataset[:, 5], beam1_name, "x", "dE")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

#Plots multiple scatter plots of the given tensors for the multiple COSY matrices.
def multiple_scatterplots(list_o_tensors, width=4, length=3):
  """
  Args:
    list_o_tensors: A list of tensors to plot (the tensors must be 2xN tensors with either x or y in each column)
    you can use loader to load the tensors in the needed configuration.
    width: The number of columns in the plot matrix.
    length: The number of rows in the plot matrix.
  """

  # Create a 4x3 plot matrix
  fig, axes = plt.subplots(width, length, figsize=(12, 16))

  # Iterate through each array and create a scatter plot on each axis
  for i in range(width):
    for j in range(length):
      # Choose the appropriate index for before_KDE based on i and j
      index = i * 3 + j
      axes[i, j].scatter(list_o_tensors[index][:, 0], list_o_tensors[index][:, 1])
      axes[i, j].set_title(f"Plot of {index}/fort.50 -> {index}/fort.51 -> {index}/fort.52")
      axes[i, j].set_xlim(-40, 40)
      axes[i, j].set_ylim(-40, 40)

  # Adjust layout and show the plots
  plt.tight_layout()
  plt.show()


#  Plots multiple KDE plots for the multiple COSY matrixes.
def multiple_KDE_plot(list_o_tensors, width=4, length=3):
  """
  Args:
    list_o_tensors: A list of tensors as produced by KDEGaussian.
    width: The number of columns in the plot matrix.
    length: The number of rows in the plot matrix.
  """

  # Create a 4x3 plot matrix
  fig, axes = plt.subplots(width, length, figsize=(12, 16))

  # Iterate through each array and create a heatmap on each axis
  for i in range(width):
    for j in range(length):
      # Choose the appropriate index for before_KDE based on i and j
      index = i * 3 + j
      sns.kdeplot(torch.squeeze(list_o_tensors[index].reshape(150, 150).detach()).T, ax=axes[i, j], cmap='coolwarm',)
      axes[i, j].set_title(f"Plot of KDE {index}/fort.50 -> {index}/fort.51 -> {index}/fort.52")

  # Adjust layout and show the plots
  plt.tight_layout()
  plt.show()


#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________


#BIG graph that has all the information one may need for a 3D matrix (MARK I).
def ALLplotting_3DMatrix_I(canonical_beam, predicted_beam, KL_storer, KL_storerA, KL_storerB, dataset_num):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
    """

    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c

    bandwidth, positions_ = bandloc(set_it=True)
    kde = KDEGaussian(bandwidth=0.005, locations=positions_)
    fig, axes = plt.subplots(7, 2, figsize=(10, 30))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    mpl.rcParams['font.size'] = 9
    
    
    coeff_array, power_array = read_map('matrix1.txt')  
    beam = transport(predicted_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[0,0], kde(loaded))
    axes[0,0].set_title(f"dataset {dataset_num} matrix A predicted_density")
    
    
    coeff_array, power_array = read_map('matrix1.txt')  
    beam = transport(canonical_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[0,1], kde(loaded))
    axes[0,1].set_title(f"dataset {dataset_num} matrix A canonical_density")
    
    
    coeff_array, power_array = read_map('matrix2.txt')  
    beam = transport(predicted_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[1,0], kde(loaded))
    axes[1,0].set_title(f"dataset {dataset_num} matrix B predicted_density")
    
    
    coeff_array, power_array = read_map('matrix2.txt')  
    beam = transport(canonical_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[1,1], kde(loaded))
    axes[1,1].set_title(f"dataset {dataset_num} matrix B canonical_density")
    
   
    axes[2,0].scatter(canonical_beam[:,0], canonical_beam[:,2], label = "canonical_beam")
    axes[2,0].scatter(predicted_beam[:,0], predicted_beam[:,2], label = "predicted_beam", marker = "x")
    axes[2,0].set_title("canonical_beam vs predicted_beam")
    axes[2,0].set_xlabel("x")
    axes[2,0].set_ylabel("y")
    b = margin_setter(predicted_beam[:,0], predicted_beam[:,2])*1.2
    axes[2,0].set_xlim(-b,b)
    axes[2,0].set_ylim(-b,b)
    axes[2,0].legend()
        
    
    axes[2,1].scatter(canonical_beam[:,0], canonical_beam[:,1], label = "canonical_beam")
    axes[2,1].scatter(predicted_beam[:,0], predicted_beam[:,1], label = "predicted_beam", marker = "x")
    axes[2,1].set_title("canonical_beam vs predicted_beam")
    axes[2,1].set_xlabel("x")
    axes[2,1].set_ylabel("aX")
    b = margin_setter(predicted_beam[:,0], predicted_beam[:,1])*1.2
    axes[2,1].set_xlim(-b,b)
    axes[2,1].set_ylim(-b,b)
    axes[2,1].legend()
    
        
    coeff_array, power_array = read_map('matrix1.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[3,0].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,2], label = "postCOSY_canon_beam")
    axes[3,0].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,2], label = "postCOSY_pred_beam", marker = "x")
    axes[3,0].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix A")
    axes[3,0].set_xlabel("x")
    axes[3,0].set_ylabel("y")
    b = 0.05
    axes[3,0].set_xlim(-b,b)
    axes[3,0].set_ylim(-b,b)
    axes[3,0].legend()
        

    coeff_array, power_array = read_map('matrix1.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[3,1].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,1], label = "postCOSY_canon_beam")
    axes[3,1].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,1], label = "postCOSY_pred_beam", marker = "x")
    axes[3,1].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix A")
    axes[3,1].set_xlabel("x")
    axes[3,1].set_ylabel("aX")
    b = margin_setter(postCOSY_pred_beam[:,0],postCOSY_pred_beam[:,1])*1.2
    axes[3,1].set_xlim(-b,b)
    axes[3,1].set_ylim(-b,b)
    axes[3,1].legend()
 
    
    coeff_array, power_array = read_map('matrix2.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[4,0].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,2], label = "postCOSY_canon_beam")
    axes[4,0].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,2], label = "postCOSY_pred_beam", marker = "x")
    axes[4,0].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix B")
    axes[4,0].set_xlabel("x")
    axes[4,0].set_ylabel("y")
    b = 0.05
    axes[4,0].set_xlim(-b,b)
    axes[4,0].set_ylim(-b,b)
    axes[4,0].legend()
    
    
    coeff_array, power_array = read_map('matrix2.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[4,1].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,1], label = "postCOSY_canon_beam")
    axes[4,1].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,1], label = "postCOSY_pred_beam", marker = "x")
    axes[4,1].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix B")
    axes[4,1].set_xlabel("x")
    axes[4,1].set_ylabel("aX")
    b = margin_setter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,1])*1.2
    axes[4,1].set_xlim(-b,b)
    axes[4,1].set_ylim(-b,b)
    axes[4,1].legend()
    
    axes[5,0].plot(range(len(KL_storer)),KL_storer)
    axes[5,0].set_yscale('log')
    axes[5,0].set_title("epoch vs Total KL")
    axes[5,0].set_xlabel("epoch")
    axes[5,0].set_ylabel("KL")
    
    axes[5,1].plot(range(len(KL_storerA)),KL_storerA)
    axes[5,1].set_yscale('log')
    axes[5,1].set_title("epoch vs KL matrix A")
    axes[5,1].set_xlabel("epoch")
    axes[5,1].set_ylabel("KL")
    
    axes[6,0].plot(range(len(KL_storerB)),KL_storerB)
    axes[6,0].set_yscale('log')
    axes[6,0].set_title("epoch vs KL matrix B")
    axes[6,0].set_xlabel("epoch")
    axes[6,0].set_ylabel("KL")

    axes[6,1].text(0.5, 0.5, "BLANK SPACE", color="black", fontsize=20, ha='center', fontweight="bold", va='center')
    axes[6,1].set_xlim(0, 1)
    axes[6,1].set_ylim(0, 1)
    axes[6,1].axis('off')
    

    plt.tight_layout()


#_____________________________________________________________________________________________________________
# _____________________________________________________________________________________________________________
# _____________________________________________________________________________________________________________    


#BIG graph that has all the information one may need for a 3D matrix (MARK II).
def ALLplotting_3DMatrix_II(canonical_beam, predicted_beam, KL_storer, KL_storerA, KL_storerB, mg_sz_storer, dataset_num):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
    """

    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c

    bandwidth, positions_ = bandloc(set_it=True)
    kde = KDEGaussian(bandwidth=0.005, locations=positions_)
    fig, axes = plt.subplots(8, 2, figsize=(10, 30))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    mpl.rcParams['font.size'] = 9
    
    
    coeff_array, power_array = read_map('matrix1.txt')  
    beam = transport(predicted_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[0,0], kde(loaded))
    axes[0,0].set_title(f"dataset {dataset_num} matrix A predicted_density")
    
    
    coeff_array, power_array = read_map('matrix1.txt')  
    beam = transport(canonical_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[0,1], kde(loaded))
    axes[0,1].set_title(f"dataset {dataset_num} matrix A canonical_density")
    
    
    coeff_array, power_array = read_map('matrix2.txt')  
    beam = transport(predicted_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[1,0], kde(loaded))
    axes[1,0].set_title(f"dataset {dataset_num} matrix B predicted_density")
    
    
    coeff_array, power_array = read_map('matrix2.txt')  
    beam = transport(canonical_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[1,1], kde(loaded))
    axes[1,1].set_title(f"dataset {dataset_num} matrix B canonical_density")
    
   
    axes[2,0].scatter(canonical_beam[:,0], canonical_beam[:,2], label = "canonical_beam")
    axes[2,0].scatter(predicted_beam[:,0], predicted_beam[:,2], label = "predicted_beam", marker = "x")
    axes[2,0].set_title("canonical_beam vs predicted_beam")
    axes[2,0].set_xlabel("x")
    axes[2,0].set_ylabel("y")
    b = margin_setter(predicted_beam[:,0], predicted_beam[:,2])*1.2
    axes[2,0].set_xlim(-b,b)
    axes[2,0].set_ylim(-b,b)
    axes[2,0].legend()
        
    
    axes[2,1].scatter(canonical_beam[:,0], canonical_beam[:,1], label = "canonical_beam")
    axes[2,1].scatter(predicted_beam[:,0], predicted_beam[:,1], label = "predicted_beam", marker = "x")
    axes[2,1].set_title("canonical_beam vs predicted_beam")
    axes[2,1].set_xlabel("x")
    axes[2,1].set_ylabel("aX")
    #b = margin_setter(predicted_beam[:,0], predicted_beam[:,1])*1.2
    b = 0.01
    axes[2,1].set_xlim(-b,b)
    axes[2,1].set_ylim(-b,b)
    axes[2,1].legend()
    
        
    coeff_array, power_array = read_map('matrix1.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[3,0].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,2], label = "postCOSY_canon_beam")
    axes[3,0].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,2], label = "postCOSY_pred_beam", marker = "x")
    axes[3,0].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix A")
    axes[3,0].set_xlabel("x")
    axes[3,0].set_ylabel("y")
    b = margin_setter(postCOSY_pred_beam[:,0],postCOSY_pred_beam[:,2])*1.2
    axes[3,0].set_xlim(-b,b)
    axes[3,0].set_ylim(-b,b)
    axes[3,0].legend()
        

    coeff_array, power_array = read_map('matrix1.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[3,1].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,1], label = "postCOSY_canon_beam")
    axes[3,1].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,1], label = "postCOSY_pred_beam", marker = "x")
    axes[3,1].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix A")
    axes[3,1].set_xlabel("x")
    axes[3,1].set_ylabel("aX")
    b = margin_setter(postCOSY_pred_beam[:,0],postCOSY_pred_beam[:,1])*1.2
    axes[3,1].set_xlim(-b,b)
    axes[3,1].set_ylim(-b,b)
    axes[3,1].legend()
 
    
    coeff_array, power_array = read_map('matrix2.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[4,0].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,2], label = "postCOSY_canon_beam")
    axes[4,0].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,2], label = "postCOSY_pred_beam", marker = "x")
    axes[4,0].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix B")
    axes[4,0].set_xlabel("x")
    axes[4,0].set_ylabel("y")
    b = 0.05
    axes[4,0].set_xlim(-b,b)
    axes[4,0].set_ylim(-b,b)
    axes[4,0].legend()
    
    
    coeff_array, power_array = read_map('matrix2.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[4,1].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,1], label = "postCOSY_canon_beam")
    axes[4,1].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,1], label = "postCOSY_pred_beam", marker = "x")
    axes[4,1].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix B")
    axes[4,1].set_xlabel("x")
    axes[4,1].set_ylabel("aX")
    b = margin_setter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,1])*1.2
    axes[4,1].set_xlim(-b,b)
    axes[4,1].set_ylim(-b,b)
    axes[4,1].legend()
    
    axes[5,0].plot(range(len(KL_storer)),KL_storer)
    axes[5,0].set_yscale('log')
    axes[5,0].set_title("epoch vs Total KL")
    axes[5,0].set_xlabel("epoch")
    axes[5,0].set_ylabel("KL")
    
    axes[5,1].plot(range(len(KL_storerA)),KL_storerA)
    axes[5,1].set_yscale('log')
    axes[5,1].set_title("epoch vs KL matrix A")
    axes[5,1].set_xlabel("epoch")
    axes[5,1].set_ylabel("KL")
    
    axes[6,0].plot(range(len(KL_storerB)),KL_storerB)
    axes[6,0].set_yscale('log')
    axes[6,0].set_title("epoch vs KL matrix B")
    axes[6,0].set_xlabel("epoch")
    axes[6,0].set_ylabel("KL")

    
    axes[6,1].plot(range(len(mg_sz_storer)),mg_sz_storer)
    axes[6,1].set_title("epoch vs margin size (it goes from - to + this value in x and y)")
    axes[6,1].set_xlabel("epoch")
    axes[6,1].set_yscale('log')
    axes[6,1].set_ylabel("margin size")

    axes[7,0].scatter(KL_storerA, KL_storerB)
    axes[7,0].set_title("KL matrix A vs KL matrix B")
    axes[7,0].set_yscale('log')
    axes[7,0].set_xscale('log')
    axes[7,0].set_xlabel("KL_A")
    axes[7,0].set_ylabel("KL_B")

    axes[7,1].scatter(mg_sz_storer[:len(KL_storer)], KL_storer)
    axes[7,1].set_title("margin size vs Total KL")
    axes[7,1].set_ylabel("total KL")
    axes[7,1].set_xlabel("margin size")
    
    plt.tight_layout()

#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________


#BIG graph that has all the information one may need for a 3D matrix (MARK III).
def ALLplotting_3DMatrix_III(canonical_beam, predicted_beam, KL_storer, KL_storerA, KL_storerB, distance_storer, dataset_num):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
    """

    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        #if c <= 0.05:
             #c = 0.05
        return c

    bandwidth, positions_ = bandloc(set_it=True)
    kde = KDEGaussian(bandwidth=0.005, locations=positions_)
    fig, axes = plt.subplots(8, 2, figsize=(10, 30))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    mpl.rcParams['font.size'] = 9
    
    coeff_array, power_array = read_map('matrix1.txt')  
    beam = transport(predicted_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[0,0], kde(loaded))
    axes[0,0].set_title(f"dataset {dataset_num} matrix A predicted_density")
    
    
    coeff_array, power_array = read_map('matrix1.txt')  
    beam = transport(canonical_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[0,1], kde(loaded))
    axes[0,1].set_title(f"dataset {dataset_num} matrix A canonical_density")
    
    
    coeff_array, power_array = read_map('matrix2.txt')  
    beam = transport(predicted_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[1,0], kde(loaded))
    axes[1,0].set_title(f"dataset {dataset_num} matrix B predicted_density")
    
    
    coeff_array, power_array = read_map('matrix2.txt')  
    beam = transport(canonical_beam, coeff_array, power_array)
    loaded = loader(beam[:,0],beam[:,2])
    KDE_plotter_ax(axes[1,1], kde(loaded))
    axes[1,1].set_title(f"dataset {dataset_num} matrix B canonical_density")
    
   
    axes[2,0].scatter(canonical_beam[:,0], canonical_beam[:,2], label = "canonical_beam")
    axes[2,0].scatter(predicted_beam[:,0], predicted_beam[:,2], label = "predicted_beam", marker = "x")
    axes[2,0].set_title("canonical_beam vs predicted_beam")
    axes[2,0].set_xlabel("x")
    axes[2,0].set_ylabel("y")
    b = margin_setter(predicted_beam[:,0], predicted_beam[:,2])*1.2
    axes[2,0].set_xlim(-b,b)
    axes[2,0].set_ylim(-b,b)
    axes[2,0].legend()
        
    
    axes[2,1].scatter(canonical_beam[:,0], canonical_beam[:,1], label = "canonical_beam")
    axes[2,1].scatter(predicted_beam[:,0], predicted_beam[:,1], label = "predicted_beam", marker = "x")
    axes[2,1].set_title("canonical_beam vs predicted_beam")
    axes[2,1].set_xlabel("x")
    axes[2,1].set_ylabel("aX")
    b = margin_setter(predicted_beam[:,0], predicted_beam[:,1])*1.2
    axes[2,1].set_xlim(-b,b)
    axes[2,1].set_ylim(-b,b)
    axes[2,1].legend()
    
        
    coeff_array, power_array = read_map('matrix1.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[3,0].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,2], label = "postCOSY_canon_beam")
    axes[3,0].scatter(postCOSY_pred_beam[:,0],postCOSY_pred_beam[:,2], label = "postCOSY_pred_beam", marker = "x")
    axes[3,0].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix A")
    axes[3,0].set_xlabel("x")
    axes[3,0].set_ylabel("y")
    b = margin_setter(postCOSY_pred_beam[:,0], postCOSY_pred_beam[:,2])*1.2
    axes[3,0].set_xlim(-b,b)
    axes[3,0].set_ylim(-b,b)
    axes[3,0].legend()
        

    coeff_array, power_array = read_map('matrix1.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[3,1].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,1], label = "postCOSY_canon_beam")
    axes[3,1].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,1], label = "postCOSY_pred_beam", marker = "x")
    axes[3,1].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix A")
    axes[3,1].set_xlabel("x")
    axes[3,1].set_ylabel("aX")
    b = margin_setter(postCOSY_pred_beam[:,0], postCOSY_pred_beam[:,1])*1.2
    axes[3,1].set_xlim(-b,b)
    axes[3,1].set_ylim(-b,b)
    axes[3,1].legend()
 
    
    coeff_array, power_array = read_map('matrix2.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[4,0].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,2], label = "postCOSY_canon_beam")
    axes[4,0].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,2], label = "postCOSY_pred_beam", marker = "x")
    axes[4,0].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix B")
    axes[4,0].set_xlabel("x")
    axes[4,0].set_ylabel("y")
    b = margin_setter(postCOSY_pred_beam[:,0], postCOSY_pred_beam[:,2])*1.2
    axes[4,0].set_xlim(-b,b)
    axes[4,0].set_ylim(-b,b)
    axes[4,0].legend()
    
    
    coeff_array, power_array = read_map('matrix2.txt')  
    postCOSY_pred_beam = transport(predicted_beam,coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam,coeff_array, power_array)
    axes[4,1].scatter(postCOSY_canon_beam[:,0],postCOSY_canon_beam[:,1], label = "postCOSY_canon_beam")
    axes[4,1].scatter(postCOSY_pred_beam [:,0],postCOSY_pred_beam[:,1], label = "postCOSY_pred_beam", marker = "x")
    axes[4,1].set_title("postCOSY_pred_beam vs postCOSY_canon_beam - matrix B")
    axes[4,1].set_xlabel("x")
    axes[4,1].set_ylabel("aX")
    b = margin_setter(postCOSY_pred_beam[:,0], postCOSY_pred_beam[:,1])*1.2
    axes[4,1].set_xlim(-b,b)
    axes[4,1].set_ylim(-b,b)
    axes[4,1].legend()
    
    axes[5,0].plot(range(len(KL_storer)),KL_storer)
    axes[5,0].set_yscale('log')
    axes[5,0].set_title("Total KL vs epoch")
    axes[5,0].set_xlabel("epoch")
    axes[5,0].set_ylabel("KL")
    
    axes[5,1].plot(range(len(KL_storerA)),KL_storerA)
    axes[5,1].set_yscale('log')
    axes[5,1].set_title("KL matrix A vs epoch")
    axes[5,1].set_xlabel("epoch")
    axes[5,1].set_ylabel("KL")
    
    axes[6,0].plot(range(len(KL_storerB)),KL_storerB)
    axes[6,0].set_yscale('log')
    axes[6,0].set_title("KL matrix B vs epoch")
    axes[6,0].set_xlabel("epoch")
    axes[6,0].set_ylabel("KL")

    axes[6,1].plot(range(len(distance_storer)),distance_storer)
    axes[6,1].set_title("distance vs epoch")
    axes[6,1].set_yscale('log')
    axes[6,1].set_xlabel("epoch")
    axes[6,1].set_ylabel("margin distance")

    axes[7,0].scatter(KL_storerA, KL_storerB)
    axes[7,0].set_title("KL matrix B vs KL matrix A")
    axes[7,0].set_xscale('log')
    axes[7,0].set_yscale('log')
    axes[7,0].set_xlabel("KL_A")
    axes[7,0].set_ylabel("KL_B")

    axes[7,1].scatter(distance_storer[:len(KL_storer)], KL_storer)
    axes[7,1].set_title("Total KL vs distance")
    axes[7,1].set_ylabel("total KL")
    axes[7,1].set_yscale('log')
    axes[7,1].set_xlabel("distance")
    
    plt.tight_layout()


#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________


# BIG graph that has all the information one may need for a 3D matrix (MARK IV).
def ALLplotting_3DMatrix_IV(canonical_beam, predicted_beam, KL_storer, KL_storerA, KL_storerB, distance_storer, dataset_num):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
        predicted_beam: training data for the beam
        KL_storer: List to store KL divergence values
        KL_storerA: List to store KL divergence values for matrix A
        KL_storerB: List to store KL divergence values for matrix B
        distance_storer: List to store the value of distance which we use to add a penalty if the simulated beam particles are outside the viewers dimensions
        dataset_num: The dataset number
    """

    # Helper function to calculate the margin
    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c
    
    # Helper function to set the range for 2D histograms
    def histogram_range(b):
        x_range = (-b, b)
        y_range = (-b, b)
        return x_range, y_range
    
    # Helper function to plot a KDE (Kernel Density Estimation) of data
    def KDE_intra_plotter(ax, rayo, matrix_number, title):
        coeff_array, power_array = read_map(f'matrix{matrix_number}.txt' ) # 'matrix1.txt'  
        beam = transport(rayo, coeff_array, power_array)
        loaded = loader(beam[:,0],beam[:,2])
        KDE_plotter_ax(ax, kde(loaded))
        ax.set_title(title)

    # Helper function to plot a 2D histogram
    def plot_2d_histogram(ax, data, x_col, y_col, title, x_label, y_label, bins=30):
        x_range, y_range = histogram_range(margin_setter(data[:, x_col], data[:, y_col]) * 1.2)
        h = ax.hist2d(data[:, x_col].numpy(), data[:, y_col].numpy(), bins=bins, cmap='coolwarm', range=[x_range, y_range])
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        cbar = plt.colorbar(h[3], ax=ax)

    # Helper function to plot the residue (difference between two 2D histograms)
    def plot_residue(axes, x_col, y_col, data1, data2, title, x_label, y_label):
        x_range, y_range = histogram_range(margin_setter(data1[:, x_col], data1[:, y_col]) * 1.2)
        hist1, xedges, yedges = np.histogram2d(data1[:, x_col].numpy(), data1[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        hist2, _, _ = np.histogram2d(data2[:, x_col].numpy(), data2[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        residue = hist1 - hist2
        residue_rotated = np.rot90(residue, k=1)
        ax = sns.heatmap(residue_rotated, cmap='coolwarm', annot=False, cbar=True, ax=axes)
        cbar = ax.collections[0].colorbar
        axes.set_title(title)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.set_xticks([])
        axes.set_yticks([])

    # Helper function to create a scatterplot
    def scatterplot_scatter(ax, x1, x2, y1, y2, label1, label2, title, x_label, y_label):
        ax.scatter(x1, x2, label=label1, s=6)
        ax.scatter(y1, y2, label=label2, marker="x", s=6)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        margin = margin_setter(y1, y2) * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.legend()

    # Helper function to plot a metric vs. epoch plot
    def plot_metric_vs_epoch(ax, data, title, x_label, y_label, y_scale='linear'):
        ax.plot(range(len(data)), data)
        ax.set_title(title)
        ax.set_yscale(y_scale)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
    # Initialize bandwidth and positions for Kernel Density Estimation (KDE)
    bandwidth, positions_ = bandloc(set_it=True) #<----- This is the values we have been using for reference (set_it is set to True)
    kde = KDEGaussian(bandwidth=0.005, locations=positions_)
    
    # Create subplots for different plots
    fig, axes = plt.subplots(9, 4, figsize=(25, 40))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    mpl.rcParams['font.size'] = 9
    
    # Plot KDE for matrix A and matrix B density predictions and real values
    KDE_intra_plotter(axes[0,0], predicted_beam, 1, f"dataset {dataset_num} matrix A predicted_density")
    KDE_intra_plotter(axes[0,1], canonical_beam, 1, f"dataset {dataset_num} matrix A canonical_density")
    KDE_intra_plotter(axes[0,2], predicted_beam, 2, f"dataset {dataset_num} matrix B predicted_density")
    KDE_intra_plotter(axes[0,3], canonical_beam, 2, f"dataset {dataset_num} matrix B canonical_density")
        
    bins = 30

    # Plot 2D histograms for various data
    plot_2d_histogram(axes[1, 0], canonical_beam, 0, 2, "canonical_beam y vs x", "x", "y", bins)
    plot_2d_histogram(axes[1, 1], canonical_beam, 0, 1, "canonical_beam aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[1, 2], predicted_beam, 0, 2, "predicted_beam y vs x", "x", "y", bins)
    plot_2d_histogram(axes[1, 3], predicted_beam, 0, 1, "predicted_beam aX vs x", "x", "aX", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[2, 0], 0, 2, canonical_beam, predicted_beam, 'Residue (canonical_beam - predicted_beam)', 'X', 'Y')
    plot_residue(axes[2, 1], 0, 1, canonical_beam, predicted_beam, 'Residue (canonical_beam - predicted_beam)', 'X', 'aX')

    # Create scatterplots to compare data
    scatterplot_scatter(axes[2, 2], canonical_beam[:, 0], canonical_beam[:, 2], predicted_beam[:, 0], predicted_beam[:, 2], "canonical_beam", "predicted_beam", "canonical_beam vs predicted_beam", "x", "y")
    scatterplot_scatter(axes[2, 3], canonical_beam[:, 0], canonical_beam[:, 1], predicted_beam[:, 0], predicted_beam[:, 1], "canonical_beam", "predicted_beam", "canonical_beam vs predicted_beam", "x", "aX")

    # Transport data using coefficient arrays and power arrays
    coeff_array, power_array = read_map('matrix1.txt')  
    postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

    # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data
    plot_2d_histogram(axes[3, 0], postCOSY_pred_beam, 0, 2, "postCOSY_pred_beam matrix 1 y vs x", "x", "y", bins)
    plot_2d_histogram(axes[3, 1], postCOSY_pred_beam, 0, 1, "postCOSY_pred_beam matrix 1 aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[3, 2], postCOSY_canon_beam, 0, 2, "postCOSY_canon_beam matrix 1 y vs x", "x", "y", bins)
    plot_2d_histogram(axes[3, 3], postCOSY_canon_beam, 0, 1, "postCOSY_canon_beam matrix 1 aX vs x", "x", "aX", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[4, 0], 0, 2, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix A (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'Y')
    plot_residue(axes[4, 1], 0, 1, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix A (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'aX')
    
    # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix A
    scatterplot_scatter(axes[4, 2], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 2], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "x", "y")
    scatterplot_scatter(axes[4, 3], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 1], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 1], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "x", "aX")

    # Transport data using coefficient arrays and power arrays for matrix B
    coeff_array, power_array = read_map('matrix2.txt')  
    postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

    # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data for matrix B
    plot_2d_histogram(axes[5, 0], postCOSY_pred_beam, 0, 2, "postCOSY_pred_beam matrix 2 y vs x", "x", "y", bins)
    plot_2d_histogram(axes[5, 1], postCOSY_pred_beam, 0, 1, "postCOSY_pred_beam matrix 2 aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[5, 2], postCOSY_canon_beam, 0, 2, "postCOSY_canon_beam matrix 2 y vs x", "x", "y", bins)
    plot_2d_histogram(axes[5, 3], postCOSY_canon_beam, 0, 1, "postCOSY_canon_beam matrix 2 aX vs x", "x", "aX", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[6, 0], 0, 2, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix B (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'Y')
    plot_residue(axes[6, 1], 0, 1, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix B (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'aX')

    # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix B
    scatterplot_scatter(axes[6, 2], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 2], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix B", "x", "y")
    scatterplot_scatter(axes[6, 3], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 1], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 1], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix B", "x", "aX")

    # Plot KL divergence, distance, and their relationships
    plot_metric_vs_epoch(axes[7, 0], KL_storer, "Total KL vs epoch", "epoch", "KL", y_scale='log')
    plot_metric_vs_epoch(axes[7, 1], KL_storerA, "KL matrix A vs epoch", "epoch", "KL", y_scale='log')
    plot_metric_vs_epoch(axes[7, 2], KL_storerB, "KL matrix B vs epoch", "epoch", "KL", y_scale='log')
    plot_metric_vs_epoch(axes[7, 3], distance_storer, "distance vs epoch", "epoch", "margin distance", y_scale='log')

    # Scatterplot of KL matrix B vs KL matrix A
    axes[8,0].scatter(KL_storerA, KL_storerB)
    axes[8,0].set_title("KL matrix B vs KL matrix A")
    axes[8,0].set_xscale('log')
    axes[8,0].set_yscale('log')
    axes[8,0].set_xlabel("KL_A")
    axes[8,0].set_ylabel("KL_B")

    # Scatterplot of Total KL vs distance
    axes[8,1].scatter(distance_storer[:len(KL_storer)], KL_storer)
    axes[8,1].set_title("Total KL vs distance")
    axes[8,1].set_ylabel("total KL")
    axes[8,1].set_yscale('log')
    axes[8,1].set_xlabel("distance")
    
    axes[8,2].text(0.5, 0.5, f"Final KL Train Loss: {min(KL_storer):.3e}", transform=axes[8, 2].transAxes, fontsize = 23, ha='center', fontweight="bold", va='center')
    axes[8,2].set_xlim(0, 1)
    axes[8,2].set_ylim(0, 1)
    axes[8,2].axis('off')

    axes[8, 3].text(0.5, 0.2, f"FinalLoss for matrix A: {min(KL_storerA):.3e}", transform=axes[8, 3].transAxes, fontsize=23, ha='center', fontweight="bold", va='center')
    axes[8, 3].text(0.5, 0.8, f"Final loss for matrix B: {min(KL_storerB):.3e}", transform=axes[8, 3].transAxes, fontsize=23, ha='center', fontweight="bold", va='center')
    axes[8, 3].set_xlim(0, 1)
    axes[8, 3].set_ylim(0, 1)
    axes[8, 3].axis('off')

    plt.tight_layout()

#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________


def ALLplotting_3DMatrix_V(canonical_beam, predicted_beam, KL_storer, KL_storerA, KL_storerB, distance_storer, dataset_num):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
        predicted_beam: training data for the beam
        KL_storer: List to store KL divergence values
        KL_storerA: List to store KL divergence values for matrix A
        KL_storerB: List to store KL divergence values for matrix B
        distance_storer: List to store the value of distance which we use to add a penalty if the simulated beam particles are outside the viewers dimensions
        dataset_num: The dataset number
    """

    # Helper function to calculate the margin
    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c
    
    # Helper function to set the range for 2D histograms
    def histogram_range(b):
        x_range = (-b, b)
        y_range = (-b, b)
        return x_range, y_range
    
    # Helper function to plot a KDE (Kernel Density Estimation) of data
    def KDE_intra_plotter(ax, rayo, matrix_number, title):
        matrix_filename = f'matrix {"ABCDEFGHIJ"[matrix_number]}.txt'
        coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
        beam = transport(rayo, coeff_array, power_array)
        loaded = loader(beam[:,0],beam[:,2])
        KDE_plotter_ax(ax, kde(loaded))
        ax.set_title(title)

    # Helper function to plot a 2D histogram
    def plot_2d_histogram(ax, data, x_col, y_col, title, x_label, y_label, bins=30):
        x_range, y_range = histogram_range(margin_setter(data[:, x_col], data[:, y_col]) * 1.2)
        h = ax.hist2d(data[:, x_col].numpy(), data[:, y_col].numpy(), bins=bins, cmap='coolwarm', range=[x_range, y_range])
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        cbar = plt.colorbar(h[3], ax=ax)

    # Helper function to plot the residue (difference between two 2D histograms)
    def plot_residue(axes, x_col, y_col, data1, data2, title, x_label, y_label):
        x_range, y_range = histogram_range(margin_setter(data1[:, x_col], data1[:, y_col]) * 1.2)
        hist1, xedges, yedges = np.histogram2d(data1[:, x_col].numpy(), data1[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        hist2, _, _ = np.histogram2d(data2[:, x_col].numpy(), data2[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        residue = hist1 - hist2
        residue_rotated = np.rot90(residue, k=1)
        ax = sns.heatmap(residue_rotated, cmap='coolwarm', annot=False, cbar=True, ax=axes)
        cbar = ax.collections[0].colorbar
        axes.set_title(title)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.set_xticks([])
        axes.set_yticks([])

    # Helper function to create a scatterplot
    def scatterplot_scatter(ax, x1, x2, y1, y2, label1, label2, title, x_label, y_label):
        ax.scatter(x1, x2, label=label1, s=6)
        ax.scatter(y1, y2, label=label2, marker="x", s=6)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        margin = margin_setter(y1, y2) * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.legend()

    # Helper function to plot a metric vs. epoch plot
    def plot_metric_vs_epoch(ax, data, title, x_label, y_label, y_scale='linear'):
        ax.plot(range(len(data)), data)
        ax.set_title(title)
        ax.set_yscale(y_scale)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
    # Initialize bandwidth and positions for Kernel Density Estimation (KDE)
    bandwidth, positions_ = bandloc(set_it=True) #<----- This is the values we have been using for reference (set_it is set to True)
    kde = KDEGaussian(bandwidth=0.005, locations=positions_)
    
    # Create subplots for different plots
    fig, axes = plt.subplots(14, 4, figsize=(30, 65))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    mpl.rcParams['font.size'] = 11
    
    # Plot KDE for matrix A and matrix B density predictions and real values
    KDE_intra_plotter(axes[0,0], predicted_beam, 1, f"dataset {dataset_num} matrix A predicted_density")
    KDE_intra_plotter(axes[0,1], canonical_beam, 1, f"dataset {dataset_num} matrix A canonical_density")
    KDE_intra_plotter(axes[0,2], predicted_beam, 2, f"dataset {dataset_num} matrix B predicted_density")
    KDE_intra_plotter(axes[0,3], canonical_beam, 2, f"dataset {dataset_num} matrix B canonical_density")
    KDE_intra_plotter(axes[1,0], predicted_beam, 3, f"dataset {dataset_num} matrix B predicted_density")
    KDE_intra_plotter(axes[1,1], canonical_beam, 3, f"dataset {dataset_num} matrix B canonical_density")
    KDE_intra_plotter(axes[1,2], predicted_beam, 4, f"dataset {dataset_num} matrix B predicted_density")
    KDE_intra_plotter(axes[1,3], canonical_beam, 4, f"dataset {dataset_num} matrix B canonical_density")
    KDE_intra_plotter(axes[2,0], predicted_beam, 5, f"dataset {dataset_num} matrix B predicted_density")
    KDE_intra_plotter(axes[2,1], canonical_beam, 5, f"dataset {dataset_num} matrix B canonical_density")
    KDE_intra_plotter(axes[2,2], predicted_beam, 6, f"dataset {dataset_num} matrix B predicted_density")
    KDE_intra_plotter(axes[2,3], canonical_beam, 6, f"dataset {dataset_num} matrix B canonical_density")

    bins = 30

    # Plot 2D histograms for various data
    plot_2d_histogram(axes[2, 0], canonical_beam, 0, 2, "canonical_beam y vs x", "x", "y", bins)
    plot_2d_histogram(axes[2, 1], canonical_beam, 0, 1, "canonical_beam aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[2, 2], predicted_beam, 0, 2, "predicted_beam y vs x", "x", "y", bins)
    plot_2d_histogram(axes[2, 3], predicted_beam, 0, 1, "predicted_beam aX vs x", "x", "aX", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[3, 0], 0, 2, canonical_beam, predicted_beam, 'Residue (canonical_beam - predicted_beam)', 'X', 'Y')
    plot_residue(axes[3, 1], 0, 1, canonical_beam, predicted_beam, 'Residue (canonical_beam - predicted_beam)', 'X', 'aX')

    # Create scatterplots to compare data
    scatterplot_scatter(axes[3, 2], canonical_beam[:, 0], canonical_beam[:, 2], predicted_beam[:, 0], predicted_beam[:, 2], "canonical_beam", "predicted_beam", "canonical_beam vs predicted_beam", "x", "y")
    scatterplot_scatter(axes[3, 3], canonical_beam[:, 0], canonical_beam[:, 1], predicted_beam[:, 0], predicted_beam[:, 1], "canonical_beam", "predicted_beam", "canonical_beam vs predicted_beam", "x", "aX")

    # Transport data using coefficient arrays and power arrays
    matrix_filename = f'matrix A.txt'
    coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
    postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

    # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data
    plot_2d_histogram(axes[4, 0], postCOSY_pred_beam, 0, 2, "postCOSY_pred_beam matrix A y vs x", "x", "y", bins)
    plot_2d_histogram(axes[4, 1], postCOSY_pred_beam, 0, 1, "postCOSY_pred_beam matrix A aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[4, 2], postCOSY_canon_beam, 0, 2, "postCOSY_canon_beam matrix A y vs x", "x", "y", bins)
    plot_2d_histogram(axes[4, 3], postCOSY_canon_beam, 0, 1, "postCOSY_canon_beam matrix A aX vs x", "x", "aX", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[5, 0], 0, 2, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix A (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'Y')
    plot_residue(axes[5, 1], 0, 1, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix A (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'aX')
    
    # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix A
    scatterplot_scatter(axes[5, 2], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 2], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "x", "y")
    scatterplot_scatter(axes[5, 3], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 1], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 1], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "x", "aX")

    # Transport data using coefficient arrays and power arrays
    matrix_filename = f'matrix B.txt'
    coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
    postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

    # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data
    plot_2d_histogram(axes[6, 0], postCOSY_pred_beam, 0, 2, "postCOSY_pred_beam matrix B y vs x", "x", "y", bins)
    plot_2d_histogram(axes[6, 1], postCOSY_pred_beam, 0, 1, "postCOSY_pred_beam matrix B aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[6, 2], postCOSY_canon_beam, 0, 2, "postCOSY_canon_beam matrix B y vs x", "x", "y", bins)
    plot_2d_histogram(axes[6, 3], postCOSY_canon_beam, 0, 1, "postCOSY_canon_beam matrix B aX vs x", "x", "aX", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[7, 0], 0, 2, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix B (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'Y')
    plot_residue(axes[7, 1], 0, 1, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix B (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'aX')
    
    # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix A
    scatterplot_scatter(axes[7, 2], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 2], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "x", "y")
    scatterplot_scatter(axes[7, 3], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 1], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 1], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "x", "aX")

    # Transport data using coefficient arrays and power arrays for matrix B
    matrix_filename = f'matrix C.txt'
    coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
    postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

    # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data for matrix B
    plot_2d_histogram(axes[8, 0], postCOSY_pred_beam, 0, 2, "postCOSY_pred_beam matrix C y vs x", "x", "y", bins)
    plot_2d_histogram(axes[8, 1], postCOSY_pred_beam, 0, 1, "postCOSY_pred_beam matrix C aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[8, 2], postCOSY_canon_beam, 0, 2, "postCOSY_canon_beam matrix C y vs x", "x", "y", bins)
    plot_2d_histogram(axes[8, 3], postCOSY_canon_beam, 0, 1, "postCOSY_canon_beam matrix C aX vs x", "x", "aX", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[9, 0], 0, 2, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix C (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'Y')
    plot_residue(axes[9, 1], 0, 1, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix C (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'aX')

    # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix B
    scatterplot_scatter(axes[9, 2], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 2], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix B", "x", "y")
    scatterplot_scatter(axes[9, 3], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 1], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 1], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix B", "x", "aX")

    # Transport data using coefficient arrays and power arrays
    matrix_filename = f'matrix D.txt'
    coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
    postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

    # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data
    plot_2d_histogram(axes[10, 0], postCOSY_pred_beam, 0, 2, "postCOSY_pred_beam matrix D y vs x", "x", "y", bins)
    plot_2d_histogram(axes[10, 1], postCOSY_pred_beam, 0, 1, "postCOSY_pred_beam matrix D aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[10, 2], postCOSY_canon_beam, 0, 2, "postCOSY_canon_beam matrix D y vs x", "x", "y", bins)
    plot_2d_histogram(axes[10, 3], postCOSY_canon_beam, 0, 1, "postCOSY_canon_beam matrix D aX vs x", "x", "aX", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[11, 0], 0, 2, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix D (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'Y')
    plot_residue(axes[11, 1], 0, 1, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix D (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'aX')
    
    # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix A
    scatterplot_scatter(axes[11, 2], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 2], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "x", "y")
    scatterplot_scatter(axes[11, 3], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 1], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 1], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "x", "aX")

    # Plot KL divergence, distance, and their relationships
    plot_metric_vs_epoch(axes[12, 0], KL_storer, "Total KL vs epoch", "epoch", "KL", y_scale='log')
    plot_metric_vs_epoch(axes[12, 1], KL_storerA, "KL matrix A vs epoch", "epoch", "KL", y_scale='log')
    plot_metric_vs_epoch(axes[12, 2], KL_storerB, "KL matrix B vs epoch", "epoch", "KL", y_scale='log')
    plot_metric_vs_epoch(axes[12, 3], distance_storer, "distance vs epoch", "epoch", "margin distance", y_scale='log')

    # Scatterplot of KL matrix B vs KL matrix A
    axes[13,0].scatter(KL_storerA, KL_storerB)
    axes[13,0].set_title("KL matrix B vs KL matrix A")
    axes[13,0].set_xscale('log')
    axes[13,0].set_yscale('log')
    axes[13,0].set_xlabel("KL_A")
    axes[13,0].set_ylabel("KL_B")

    # Scatterplot of Total KL vs distance
    axes[13,1].scatter(distance_storer[:len(KL_storer)], KL_storer)
    axes[13,1].set_title("Total KL vs distance")
    axes[13,1].set_ylabel("total KL")
    axes[13,1].set_yscale('log')
    axes[13,1].set_xlabel("distance")
    
    axes[13,2].text(0.5, 0.5, f"Final KL Train Loss: {min(KL_storer):.3e}", transform=axes[13, 2].transAxes, fontsize = 23, ha='center', fontweight="bold", va='center')
    axes[13,2].set_xlim(0, 1)
    axes[13,2].set_ylim(0, 1)
    axes[13,2].axis('off')

    axes[13,3].text(0.5, 0.2, f"FinalLoss for matrix A: {min(KL_storerA):.3e}", transform=axes[13,3].transAxes, fontsize=23, ha='center', fontweight="bold", va='center')
    axes[13,3].text(0.5, 0.8, f"Final loss for matrix B: {min(KL_storerB):.3e}", transform=axes[13,3].transAxes, fontsize=23, ha='center', fontweight="bold", va='center')
    axes[13,3].set_xlim(0, 1)
    axes[13,3].set_ylim(0, 1)
    axes[13,3].axis('off')

    plt.tight_layout()

#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________

# Helper function to calculate the margin
def ALLplotting_3DMatrix_variable_number_of_matrices(canonical_beam, predicted_beam, KL_individual_storers, bandwidth, number_of_matrices, dataset_num, bins):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
        predicted_beam: training data for the beam
        KL_storer: List to store KL divergence values
        KL_storerA: List to store KL divergence values for matrix A
        KL_storerB: List to store KL divergence values for matrix B
        distance_storer: List to store the value of distance which we use to add a penalty if the simulated beam particles are outside the viewers dimensions
        dataset_num: The dataset number
    """
    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c

    # Helper function to set the range for 2D histograms
    def histogram_range(b):
        x_range = (-b, b)
        y_range = (-b, b)
        return x_range, y_range

    # Helper function to plot a KDE (Kernel Density Estimation) of data
    def KDE_intra_plotter(ax, beam, title):
        loaded = loader(beam[:,0],beam[:,2])
        KDE_plotter_ax(ax, kde(loaded))
        ax.set_title(title)

    # Helper function to plot a 2D histogram
    def plot_2d_histogram(ax, data, x_col, y_col, title, x_label, y_label, bins=30):
        x_range, y_range = histogram_range(margin_setter(data[:, x_col], data[:, y_col]) * 1.2)
        h = ax.hist2d(data[:, x_col].numpy(), data[:, y_col].numpy(), bins=bins, cmap='coolwarm', range=[x_range, y_range])
        ax.set_title(title)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        cbar = plt.colorbar(h[3], ax=ax)

    # Helper function to plot the residue (difference between two 2D histograms)
    def plot_residue(axes, x_col, y_col, data1, data2, title, x_label, y_label, bins):
        x_range, y_range = histogram_range(margin_setter(data1[:, x_col], data1[:, y_col]) * 1.2)
        hist1, xedges, yedges = np.histogram2d(data1[:, x_col].numpy(), data1[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        hist2, _, _ = np.histogram2d(data2[:, x_col].numpy(), data2[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        residue = hist1 - hist2
        residue_rotated = np.rot90(residue, k=1)
        ax = sns.heatmap(residue_rotated, cmap='coolwarm', annot=False, cbar=True, ax=axes)
        cbar = ax.collections[0].colorbar
        axes.set_title(title, fontsize=21)
        axes.set_xlabel(x_label, fontsize=22)
        axes.set_ylabel(y_label, fontsize=22)
        axes.set_xticks([])
        axes.set_yticks([])

    # Helper function to create a scatterplot
    def scatterplot_scatter(ax, x1, x2, y1, y2, label1, label2, title, x_label, y_label):
        ax.scatter(x1, x2, label=label1, s=6)
        ax.scatter(y1, y2, label=label2, marker="x", s=6)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        margin = margin_setter(y1, y2) * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.legend()

    # Helper function to plot a metric vs. epoch plot
    def plot_metric_vs_epoch(ax, data, title, x_label, y_label, y_scale='linear'):
        ax.plot(range(len(data)), data)
        ax.set_title(title)
        ax.set_yscale(y_scale)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)

    #This function prints the lowest KL for a given matrix
    def display_lowest_loss(ax, loss_storer, beam_origin):
        ax.text(0.5, 0.5, f"Lowest loss for {beam_origin}: {min(loss_storer):.3e}", transform=ax.transAxes, fontsize=25, ha='center', fontweight="bold", va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')    


    def function(beam1, beam2, loss_storer, beam1_name, beam2_name, bins, beam_origin, index):
        # Plot KDE for current matrix
        KDE_intra_plotter(axes[3*index, 0], beam1, f"dataset {dataset_num} {beam_origin} {beam1_name}")
        KDE_intra_plotter(axes[3*index, 1], beam2, f"dataset {dataset_num} {beam_origin} {beam2_name}")

        # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data
        plot_2d_histogram(axes[3*index, 2], beam1, 0, 2, f"{beam1_name} {beam_origin} y vs x", "x", "y", bins)
        plot_2d_histogram(axes[3*index, 3], beam1, 0, 1, f"{beam1_name} {beam_origin} aX vs x", "x", "aX", bins)
        plot_2d_histogram(axes[3*index+1, 0], beam2, 0, 2, f"{beam2_name} {beam_origin} y vs x", "x", "y", bins)
        plot_2d_histogram(axes[3*index+1, 1], beam2, 0, 1, f"{beam2_name} {beam_origin} aX vs x", "x", "aX", bins)

        # Plot residue (difference) between canonical_beam and predicted_beam
        plot_residue(axes[3*index+1, 2], 0, 2, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'Y', bins)
        plot_residue(axes[3*index+1, 3], 0, 1, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'aX', bins)

        # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix A
        scatterplot_scatter(axes[3*index+2, 0], beam1[:, 0], beam1[:, 2], beam2[:, 0], beam2[:, 2], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name} - {beam_origin}", "x", "y")
        scatterplot_scatter(axes[3*index+2, 1], beam1[:, 0], beam1[:, 1], beam2[:, 0], beam2[:, 1], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name} - {beam_origin}", "x", "aX")

        # Plot KL divergence, distance, and their relationships
        plot_metric_vs_epoch(axes[3*index+2, 2], loss_storer[index], f"Loss {beam_origin} vs epoch", "epoch", "KL", y_scale='log')
        
        # This prints the lowest matrix for the given matrix
        display_lowest_loss(axes[3*index+2, 3], loss_storer[index], beam_origin)

        
    # Initialize bandwidth and positions for Kernel Density Estimation (KDE)
    positions_ = bandloc(set_it=True) #<----- This is the values we have been using for reference (set_it is set to True)
    kde = KDEGaussian(bandwidth=0.005, locations=positions_)
    
    # Create subplots for different plots
    fig, axes = plt.subplots(3*(number_of_matrices+1), 4, figsize=(46, 24*(number_of_matrices+1)))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    mpl.rcParams['font.size'] = 25
    

    function(predicted_beam, canonical_beam, KL_individual_storers,"predicted_beam", "canonical_beam", bins, "", 0)
        
    for index in range(number_of_matrices):
        # Current matrix letter
        current_letter = "ABCDEFGHIJ"[index]

        # Transport data using coefficient arrays and power arrays
        matrix_filename = f'matrix {current_letter}.txt'
        coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
        postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
        postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

        
        function(postCOSY_pred_beam, postCOSY_canon_beam, KL_individual_storers,"pCOSY_pred_beam", "pCOSY_canon_beam",bins ,f'matrix {current_letter}', index+1)

    

#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________


# Plotter function to plot the results of training independent of the number of matrices used
def ALLplotting_5DMatrix_variable_number_of_matrices(canonical_beam, predicted_beam, KL_individual_storers, bandwidth, number_of_matrices, dataset_num, bins):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
        predicted_beam: training data for the beam
        KL_storer: List to store KL divergence values
        KL_storerA: List to store KL divergence values for matrix A
        KL_storerB: List to store KL divergence values for matrix B
        distance_storer: List to store the value of distance which we use to add a penalty if the simulated beam particles are outside the viewers dimensions
        dataset_num: The dataset number
    """
    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c

    # Helper function to set the range for 2D histograms
    def histogram_range(b):
        x_range = (-b, b)
        y_range = (-b, b)
        return x_range, y_range

    # Helper function to plot a KDE (Kernel Density Estimation) of data
    def KDE_intra_plotter(ax, beam, title):
        loaded = loader(beam[:,0],beam[:,2])
        KDE_plotter_ax(ax, kde(loaded))
        ax.set_title(title)

    # Helper function to plot a 2D histogram
    def plot_2d_histogram(ax, data, x_col, y_col, title, x_label, y_label, bins=30):
        x_range, y_range = histogram_range(margin_setter(data[:, x_col], data[:, y_col]) * 1.2)
        h = ax.hist2d(data[:, x_col].numpy(), data[:, y_col].numpy(), bins=bins, cmap='coolwarm', range=[x_range, y_range])
        ax.set_title(title)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        cbar = plt.colorbar(h[3], ax=ax)

    # Helper function to plot the residue (difference between two 2D histograms)
    def plot_residue(axes, x_col, y_col, data1, data2, title, x_label, y_label, bins):
        x_range, y_range = histogram_range(margin_setter(data1[:, x_col], data1[:, y_col]) * 1.2)
        hist1, xedges, yedges = np.histogram2d(data1[:, x_col].numpy(), data1[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        hist2, _, _ = np.histogram2d(data2[:, x_col].numpy(), data2[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        residue = hist1 - hist2
        residue_rotated = np.rot90(residue, k=1)
        ax = sns.heatmap(residue_rotated, cmap='coolwarm', annot=False, cbar=True, ax=axes)
        cbar = ax.collections[0].colorbar
        axes.set_title(title, fontsize=21)
        axes.set_xlabel(x_label, fontsize=22)
        axes.set_ylabel(y_label, fontsize=22)
        axes.set_xticks([])
        axes.set_yticks([])

    # Helper function to create a scatterplot
    def scatterplot_scatter(ax, x1, x2, y1, y2, label1, label2, title, x_label, y_label):
        ax.scatter(x1, x2, label=label1, s=6)
        ax.scatter(y1, y2, label=label2, marker="x", s=6)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        margin = margin_setter(y1, y2) * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.legend()

    # Helper function to plot a metric vs. epoch plot
    def plot_metric_vs_epoch(ax, data, title, x_label, y_label, y_scale='linear'):
        ax.plot(range(len(data)), data)
        ax.set_title(title)
        ax.set_yscale(y_scale)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)

    #This function prints the lowest KL for a given matrix
    def display_lowest_loss(ax, loss_storer, beam_origin):
        ax.text(0.5, 0.5, f"Lowest loss for {beam_origin}: {min(loss_storer):.3e}", transform=ax.transAxes, fontsize=25, ha='center', fontweight="bold", va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')    


    def function(beam1, beam2, loss_storer, beam1_name, beam2_name, bins, beam_origin, index):
        # Plot KDE for current matrix
        KDE_intra_plotter(axes[5*index, 0], beam1, f"dataset {dataset_num} {beam_origin} {beam1_name}")
        KDE_intra_plotter(axes[5*index, 1], beam2, f"dataset {dataset_num} {beam_origin} {beam2_name}")

        # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data
        plot_2d_histogram(axes[5*index, 2], beam1, 0, 2, f"{beam1_name} {beam_origin} y vs x", "x", "y", bins)
        plot_2d_histogram(axes[5*index, 3], beam1, 0, 1, f"{beam1_name} {beam_origin} aX vs x", "x", "aX", bins)
        plot_2d_histogram(axes[5*index+1, 0], beam1, 2, 3, f"{beam1_name} {beam_origin} aY vs y", "y", "aY", bins)
        plot_2d_histogram(axes[5*index+1, 1], beam1, 0, 5, f"{beam1_name} {beam_origin} dE vs X", "dE", "X", bins)
        plot_2d_histogram(axes[5*index+1, 2], beam2, 0, 2, f"{beam2_name} {beam_origin} y vs x", "x", "y", bins)
        plot_2d_histogram(axes[5*index+1, 3], beam2, 0, 1, f"{beam2_name} {beam_origin} aX vs x", "x", "aX", bins)
        plot_2d_histogram(axes[5*index+2, 0], beam2, 2, 3, f"{beam1_name} {beam_origin} aY vs y", "y", "aY", bins)
        plot_2d_histogram(axes[5*index+2, 1], beam2, 0, 5, f"{beam1_name} {beam_origin} dE vs X", "dE", "X", bins)

        # Plot residue (difference) between canonical_beam and predicted_beam
        plot_residue(axes[5*index+2, 2], 0, 2, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'Y', bins)
        plot_residue(axes[5*index+2, 3], 0, 1, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'aX', bins)
        plot_residue(axes[5*index+3, 0], 2, 3, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'Y', 'aY', bins)
        plot_residue(axes[5*index+3, 1], 0, 5, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'dE', bins)

        # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix A
        scatterplot_scatter(axes[5*index+3, 2], beam1[:, 0], beam1[:, 2], beam2[:, 0], beam2[:, 2], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name} - {beam_origin}", "x", "y")
        scatterplot_scatter(axes[5*index+3, 3], beam1[:, 0], beam1[:, 1], beam2[:, 0], beam2[:, 1], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name} - {beam_origin}", "x", "aX")
        scatterplot_scatter(axes[5*index+4, 0], beam1[:, 2], beam1[:, 3], beam2[:, 2], beam2[:, 3], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name} - {beam_origin}", "y", "aY")
        scatterplot_scatter(axes[5*index+4, 1], beam1[:, 0], beam1[:, 5], beam2[:, 0], beam2[:, 5], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name} - {beam_origin}", "x", "dE")

        # Plot KL divergence, distance, and their relationships
        plot_metric_vs_epoch(axes[5*index+4, 2], loss_storer[index], f"Loss {beam_origin} vs epoch", "epoch", "KL", y_scale='log')
        
        # This prints the lowest matrix for the given matrix
        display_lowest_loss(axes[5*index+4, 3], loss_storer[index], beam_origin)

        
    # Initialize bandwidth and positions for Kernel Density Estimation (KDE)
    positions_ = bandloc(set_it=True) #<----- This is the values we have been using for reference (set_it is set to True)
    kde = KDEGaussian(bandwidth=bandwidth, locations=positions_)
    
    # Create subplots for different plots
    fig, axes = plt.subplots(5*(number_of_matrices+1), 4, figsize=(46, 40*(number_of_matrices+1)))#figsize=(23, 80))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    mpl.rcParams['font.size'] = 25
    #fig, axes = plt.subplots(5*(number_of_matrices+1), 4, figsize=(6, 30))#figsize=(23, 80))
    #plt.subplots_adjust(hspace=0.3, wspace=0.3)
       
  
    function(predicted_beam, canonical_beam, KL_individual_storers,"predicted_beam", "canonical_beam", bins, "", 0)
        
    for index in range(number_of_matrices):
        # Current matrix letter
        current_letter = "ABCDEFGHIJ"[index]

        # Transport data using coefficient arrays and power arrays
        matrix_filename = f'matrix {current_letter}.txt'
        coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
        postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
        postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

        
        function(postCOSY_pred_beam, postCOSY_canon_beam, KL_individual_storers,"pCOSY_pred_beam", "pCOSY_canon_beam",bins ,f'matrix {current_letter}', index+1)
    


#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
        

# Plotter function to plot the results of training independent of the number of matrices used
def ALLplotting_5DMatrix_variable_number_of_matrices_2(canonical_beam, predicted_beam, KL_individual_storers, bandwidth, number_of_matrices, dataset_num, bins):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
        predicted_beam: training data for the beam
        KL_storer: List to store KL divergence values
        KL_storerA: List to store KL divergence values for matrix A
        KL_storerB: List to store KL divergence values for matrix B
        distance_storer: List to store the value of distance which we use to add a penalty if the simulated beam particles are outside the viewers dimensions
        dataset_num: The dataset number
    """
    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c

    # Helper function to set the range for 2D histograms
    def histogram_range(b):
        x_range = (-b, b)
        y_range = (-b, b)
        return x_range, y_range

    # Helper function to plot a KDE (Kernel Density Estimation) of data
    def KDE_intra_plotter(ax, beam, title):
        loaded = loader(beam[:,0],beam[:,2])
        KDE_plotter_ax(ax, kde(loaded))
        ax.set_title(title)

    # Helper function to plot a 2D histogram
    def plot_2d_histogram(ax, data, x_col, y_col, title, x_label, y_label, bins=30):
        x_range, y_range = histogram_range(margin_setter(data[:, x_col], data[:, y_col]) * 1.2)
        h = ax.hist2d(data[:, x_col].numpy(), data[:, y_col].numpy(), bins=bins, cmap='coolwarm', range=[x_range, y_range])
        ax.set_title(title)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        cbar = plt.colorbar(h[3], ax=ax)

    # Helper function to plot the residue (difference between two 2D histograms)
    def plot_residue(axes, x_col, y_col, data1, data2, title, x_label, y_label, bins):
        x_range, y_range = histogram_range(margin_setter(data1[:, x_col], data1[:, y_col]) * 1.2)
        hist1, xedges, yedges = np.histogram2d(data1[:, x_col].numpy(), data1[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        hist2, _, _ = np.histogram2d(data2[:, x_col].numpy(), data2[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        residue = hist1 - hist2
        residue_rotated = np.rot90(residue, k=1)
        ax = sns.heatmap(residue_rotated, cmap='coolwarm', annot=False, cbar=True, ax=axes)
        cbar = ax.collections[0].colorbar
        axes.set_title(title, fontsize=21)
        axes.set_xlabel(x_label, fontsize=22)
        axes.set_ylabel(y_label, fontsize=22)
        axes.set_xticks([])
        axes.set_yticks([])

    # Helper function to create a scatterplot
    def scatterplot_scatter(ax, x1, x2, y1, y2, label1, label2, title, x_label, y_label):
        ax.scatter(y1, y2, label=label2, marker="x", s=6)
        ax.scatter(x1, x2, label=label1, s=6)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        margin = margin_setter(y1, y2) * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.legend()

    # Helper function to plot a metric vs. epoch plot
    def plot_metric_vs_epoch(ax, data, title, x_label, y_label, y_scale='linear', index = None):
        if len(data) == 3 and index is not None:
            ax.plot(range(len(data[0][index])), data[0][index], label = "total loss", color = "green")
            ax.plot(range(len(data[1][index])), data[1][index], label = "kl", color = "blue")
            ax.plot(range(len(data[2][index])), data[2][index], label = "distance", color = "red")
            ax.set_title(title)
            ax.set_yscale(y_scale)
            ax.set_xlabel(x_label, fontsize=22)
            ax.set_ylabel(y_label, fontsize=22)
            ax.legend()
        else:
            ax.plot(range(len(data)), data)
            ax.set_title(title)
            ax.set_yscale(y_scale)
            ax.set_xlabel(x_label, fontsize=22)
            ax.set_ylabel(y_label, fontsize=22)


    #This function prints the lowest KL for a given matrix
    def display_lowest_loss(ax, loss_storer, beam_origin, index  = None):
        if index is None:
            index = 0
            
        if len(loss_storer) == 3:
            ax.text(0.5, 0.5, f"Lowest loss for {beam_origin}: {min(loss_storer[0][index]):.3e}", transform=ax.transAxes, fontsize=25, ha='center', fontweight="bold", va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')    
        else:
            ax.text(0.5, 0.5, f"Lowest loss for {beam_origin}: {min(loss_storer):.3e}", transform=ax.transAxes, fontsize=25, ha='center', fontweight="bold", va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')  
              

    def find_number_of_rows(number):
        for index in range(number):
            # Calculate row and column indices dynamically
            row_index = 5 + (index - 1) // 2
            col_index = 2 * ((index - 1) % 2)

        last_col_index = col_index
        last_row_index = row_index

        for index in range (1 + number):
            # Calculate row and column indices dynamically
            if (last_col_index + 1) == 3:
                row_index = last_row_index + (index) // 4
                row_index = row_index + 1 
                col_index = (index) % 4
            else:
                row_index = last_row_index + (index + 2) // 4
                col_index = (index + 2) % 4

        return row_index+1
        
    # Initialize bandwidth and positions for Kernel Density Estimation (KDE)
    positions_ = bandloc(set_it=True) #<----- This is the values we have been using for reference (set_it is set to True)
    kde = KDEGaussian(bandwidth=bandwidth, locations=positions_)
    
    # Create subplots for different plots
    n_rows = find_number_of_rows(number_of_matrices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(46, 8*n_rows))#figsize=(23, 80))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    mpl.rcParams['font.size'] = 25
    #fig, axes = plt.subplots(5*(number_of_matrices+1), 4, figsize=(6, 30))#figsize=(23, 80))
    #plt.subplots_adjust(hspace=0.3, wspace=0.3)
       

    beam1 = predicted_beam
    beam2 = canonical_beam
    beam1_name = "predicted_beam"
    beam2_name = "canonical_beam"

    scatterplot_scatter(axes[0, 0], beam1[:, 0], beam1[:, 2], beam2[:, 0], beam2[:, 2], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "y")
    scatterplot_scatter(axes[0, 1], beam1[:, 0], beam1[:, 1], beam2[:, 0], beam2[:, 1], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "aX")
    scatterplot_scatter(axes[0, 2], beam1[:, 2], beam1[:, 3], beam2[:, 2], beam2[:, 3], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "y", "aY")
    scatterplot_scatter(axes[0, 3], beam1[:, 0], beam1[:, 5], beam2[:, 0], beam2[:, 5], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "dE")

    plot_2d_histogram(axes[1, 0], beam2, 0, 2, f"{beam2_name} y vs x", "x", "y", bins)
    plot_2d_histogram(axes[1, 1], beam2, 0, 1, f"{beam2_name} aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[1, 2], beam2, 2, 3, f"{beam2_name} aY vs y", "y", "aY", bins)
    plot_2d_histogram(axes[1, 3], beam2, 0, 5, f"{beam2_name} dE vs X", "dE", "X", bins)

    plot_2d_histogram(axes[2, 0], beam1, 0, 2, f"{beam1_name} y vs x", "x", "y", bins)
    plot_2d_histogram(axes[2, 1], beam1, 0, 1, f"{beam1_name} aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[2, 2], beam1, 2, 3, f"{beam1_name} aY vs y", "y", "aY", bins)
    plot_2d_histogram(axes[2, 3], beam1, 0, 5, f"{beam1_name} dE vs X", "dE", "X", bins)

    
    plot_residue(axes[3, 0], 0, 2, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'Y', bins)
    plot_residue(axes[3, 1], 0, 1, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'aX', bins)
    plot_residue(axes[3, 2], 2, 3, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'Y', 'aY', bins)
    plot_residue(axes[3, 3], 0, 5, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'dE', bins)

    KDE_intra_plotter(axes[4, 0], beam1, f"dataset {dataset_num} {beam1_name}")
    KDE_intra_plotter(axes[4, 1], beam2, f"dataset {dataset_num} {beam2_name}")


    for index in range(number_of_matrices):
        # Current matrix letter
        current_letter = "ABCDEFGHIJ"[index]

        # Transport data using coefficient arrays and power arrays
        matrix_filename = f'matrix {current_letter}.txt'
        coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
        beam1= transport(predicted_beam, coeff_array, power_array)
        beam2 = transport(canonical_beam, coeff_array, power_array)
        beam_origin = f'matrix {current_letter}'

        if index == 0:
            KDE_intra_plotter(axes[4, 2], beam1, f"dataset {dataset_num} {beam_origin} {beam1_name}")
            KDE_intra_plotter(axes[4, 3], beam2, f"dataset {dataset_num} {beam_origin} {beam2_name}")

        else:
            row_index = 5 + (index - 1) // 2
            col_index = 2 * ((index - 1) % 2)
            KDE_intra_plotter(axes[row_index, col_index], beam2, f"dataset {dataset_num} {beam_origin} {beam2_name}")
            KDE_intra_plotter(axes[row_index, col_index+1], beam1, f"dataset {dataset_num} {beam_origin} {beam1_name}")
        
    last_col_index = col_index
    last_row_index = row_index

    for index in range(len(KL_individual_storers[0])):
        # create plot name dynamicaly
        if index == 0:
            title = "Total loss vs epoch"
        else:
            title = f"loss for matrix {index+1} 15 Feb vs epoch"

        # Calculate row and column indices dynamically
        if (last_col_index + 1) == 3:
            row_index = last_row_index + (index) // 4
            row_index = row_index + 1 
            col_index = (index) % 4
        else:
            row_index = last_row_index + (index + 2) // 4
            col_index = (index + 2) % 4

        plot_metric_vs_epoch(axes[row_index, col_index], KL_individual_storers, title, "epoch", "loss", y_scale='log', index=index-1)

    display_lowest_loss(axes[row_index, col_index+1], KL_individual_storers, "for whole model")

    for i in range(2):
        for j in range(4):
            current_ax = axes[n_rows - 2 + i, j]
            
            # Check if the content in the current axes is created by display_lowest_loss
            if not current_ax.lines and not current_ax.texts:
                fig.delaxes(current_ax)


#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
        

# Plotter function to plot the results of training independent of the number of matrices used
def ALLplotting_5DMatrix_variable_number_of_matrices_3(canonical_beam, predicted_beam, KL_individual_storers, bandwidth, number_of_matrices, dataset_num, bins):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
        predicted_beam: training data for the beam
        KL_storer: List to store KL divergence values
        KL_storerA: List to store KL divergence values for matrix A
        KL_storerB: List to store KL divergence values for matrix B
        distance_storer: List to store the value of distance which we use to add a penalty if the simulated beam particles are outside the viewers dimensions
        dataset_num: The dataset number
    """
    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c

    # Helper function to set the range for 2D histograms
    def histogram_range(b):
        x_range = (-b, b)
        y_range = (-b, b)
        return x_range, y_range

    # Helper function to plot a KDE (Kernel Density Estimation) of data
    def KDE_intra_plotter(ax, beam, title):
        loaded = loader(beam[:,0],beam[:,2])
        KDE_plotter_ax(ax, kde(loaded))
        ax.set_title(title)

    # Helper function to plot a 2D histogram
    def plot_2d_histogram(ax, data, x_col, y_col, title, x_label, y_label, bins=30):
        x_range, y_range = histogram_range(margin_setter(data[:, x_col], data[:, y_col]) * 1.2)
        h = ax.hist2d(data[:, x_col].numpy(), data[:, y_col].numpy(), bins=bins, cmap='coolwarm', range=[x_range, y_range])
        ax.set_title(title)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        cbar = plt.colorbar(h[3], ax=ax)

    # Helper function to plot the residue (difference between two 2D histograms)
    def plot_residue(axes, x_col, y_col, data1, data2, title, x_label, y_label, bins):
        x_range, y_range = histogram_range(margin_setter(data1[:, x_col], data1[:, y_col]) * 1.2)
        hist1, xedges, yedges = np.histogram2d(data1[:, x_col].numpy(), data1[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        hist2, _, _ = np.histogram2d(data2[:, x_col].numpy(), data2[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        residue = hist1 - hist2
        residue_rotated = np.rot90(residue, k=1)
        ax = sns.heatmap(residue_rotated, cmap='coolwarm', annot=False, cbar=True, ax=axes)
        cbar = ax.collections[0].colorbar
        axes.set_title(title, fontsize=21)
        axes.set_xlabel(x_label, fontsize=22)
        axes.set_ylabel(y_label, fontsize=22)
        axes.set_xticks([])
        axes.set_yticks([])

    # Helper function to create a scatterplot
    def scatterplot_scatter(ax, x1, x2, y1, y2, label1, label2, title, x_label, y_label):
        ax.scatter(y1, y2, label=label2, marker="x", s=6)
        ax.scatter(x1, x2, label=label1, s=6)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        margin = margin_setter(y1, y2) * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.legend()

    # Helper function to plot a metric vs. epoch plot
    def plot_metric_vs_epoch(ax, data, title, x_label, y_label, y_scale='linear', index = None):
        if len(data) == 3 and index is not None:
            ax.plot(range(len(data[0][index])), data[0][index], label = "total loss", color = "green")
            ax.plot(range(len(data[1][index])), data[1][index], label = "kl", color = "blue")
            ax.plot(range(len(data[2][index])), data[2][index], label = "distance", color = "red")
            ax.set_title(title)
            ax.set_yscale(y_scale)
            ax.set_xlabel(x_label, fontsize=22)
            ax.set_ylabel(y_label, fontsize=22)
            ax.legend()
        else:
            ax.plot(range(len(data)), data)
            ax.set_title(title)
            ax.set_yscale(y_scale)
            ax.set_xlabel(x_label, fontsize=22)
            ax.set_ylabel(y_label, fontsize=22)


    #This function prints the lowest KL for a given matrix
    def display_lowest_loss(ax, loss_storer, beam_origin, index  = None):
        if index is None:
            index = 0
            
        if len(loss_storer) == 3:
            ax.text(0.5, 0.5, f"Lowest loss for {beam_origin}: {min(loss_storer[0][index]):.3e}", transform=ax.transAxes, fontsize=25, ha='center', fontweight="bold", va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')    
        else:
            ax.text(0.5, 0.5, f"Lowest loss for {beam_origin}: {min(loss_storer):.3e}", transform=ax.transAxes, fontsize=25, ha='center', fontweight="bold", va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')  
              

    def find_number_of_rows(number):
        for index in range(number):
            # Calculate row and column indices dynamically
            row_index = 5 + (index - 1) // 2
            col_index = 2 * ((index - 1) % 2)

        last_col_index = col_index
        last_row_index = row_index

        for index in range (1 + number):
            # Calculate row and column indices dynamically
            if (last_col_index + 1) == 3:
                row_index = last_row_index + (index) // 4
                row_index = row_index + 1 
                col_index = (index) % 4
            else:
                row_index = last_row_index + (index + 2) // 4
                col_index = (index + 2) % 4

        return row_index+1
        
    # Initialize bandwidth and positions for Kernel Density Estimation (KDE)
    positions_ = bandloc(set_it=True) #<----- This is the values we have been using for reference (set_it is set to True)
    kde = KDEGaussian(bandwidth=bandwidth, locations=positions_)
    
    # Create subplots for different plots
    n_rows = find_number_of_rows(number_of_matrices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(46, 8*n_rows))#figsize=(23, 80))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    mpl.rcParams['font.size'] = 25
    #fig, axes = plt.subplots(5*(number_of_matrices+1), 4, figsize=(6, 30))#figsize=(23, 80))
    #plt.subplots_adjust(hspace=0.3, wspace=0.3)
       

    beam1 = predicted_beam
    beam2 = canonical_beam
    beam1_name = "predicted_beam"
    beam2_name = "canonical_beam"

    scatterplot_scatter(axes[0, 0], beam1[:, 0], beam1[:, 2], beam2[:, 0], beam2[:, 2], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "y")
    scatterplot_scatter(axes[0, 1], beam1[:, 0], beam1[:, 1], beam2[:, 0], beam2[:, 1], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "aX")
    scatterplot_scatter(axes[0, 2], beam1[:, 2], beam1[:, 3], beam2[:, 2], beam2[:, 3], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "y", "aY")
    scatterplot_scatter(axes[0, 3], beam1[:, 0], beam1[:, 5], beam2[:, 0], beam2[:, 5], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "dE")

    plot_2d_histogram(axes[1, 0], beam2, 0, 2, f"{beam2_name} y vs x", "x", "y", bins)
    plot_2d_histogram(axes[1, 1], beam2, 0, 1, f"{beam2_name} aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[1, 2], beam2, 2, 3, f"{beam2_name} aY vs y", "y", "aY", bins)
    plot_2d_histogram(axes[1, 3], beam2, 0, 5, f"{beam2_name} dE vs X", "dE", "X", bins)

    plot_2d_histogram(axes[2, 0], beam1, 0, 2, f"{beam1_name} y vs x", "x", "y", bins)
    plot_2d_histogram(axes[2, 1], beam1, 0, 1, f"{beam1_name} aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[2, 2], beam1, 2, 3, f"{beam1_name} aY vs y", "y", "aY", bins)
    plot_2d_histogram(axes[2, 3], beam1, 0, 5, f"{beam1_name} dE vs X", "dE", "X", bins)

    
    plot_residue(axes[3, 0], 0, 2, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'Y', bins)
    plot_residue(axes[3, 1], 0, 1, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'aX', bins)
    plot_residue(axes[3, 2], 2, 3, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'Y', 'aY', bins)
    plot_residue(axes[3, 3], 0, 5, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'dE', bins)

    KDE_intra_plotter(axes[4, 0], beam1, f"dataset {dataset_num} {beam1_name}")
    KDE_intra_plotter(axes[4, 1], beam2, f"dataset {dataset_num} {beam2_name}")


    for index in range(number_of_matrices):
        matrix_filename = f'Matrix {index+1} 15 Feb.txt'
        coeff_array, power_array = read_map(f'./matrices2/{matrix_filename}')   

        beam1= transport(predicted_beam, coeff_array, power_array)
        beam2 = transport(canonical_beam, coeff_array, power_array)
        beam_origin = f'Matrix {index+1} 15 Feb'

        if index == 0:
            KDE_intra_plotter(axes[4, 2], beam1, f"dataset {dataset_num} {beam_origin} {beam1_name}")
            KDE_intra_plotter(axes[4, 3], beam2, f"dataset {dataset_num} {beam_origin} {beam2_name}")

        else:
            row_index = 5 + (index - 1) // 2
            col_index = 2 * ((index - 1) % 2)
            KDE_intra_plotter(axes[row_index, col_index], beam2, f"dataset {dataset_num} {beam_origin} {beam2_name}")
            KDE_intra_plotter(axes[row_index, col_index+1], beam1, f"dataset {dataset_num} {beam_origin} {beam1_name}")
        
    last_col_index = col_index
    last_row_index = row_index

    for index in range(len(KL_individual_storers[0])):
        # create plot name dynamicaly
        if index == 0:
            title = "Total loss vs epoch"
        else:
            title = f"loss for matrix {index+1} 15 Feb vs epoch"

        # Calculate row and column indices dynamically
        if (last_col_index + 1) == 3:
            row_index = last_row_index + (index) // 4
            row_index = row_index + 1 
            col_index = (index) % 4
        else:
            row_index = last_row_index + (index + 2) // 4
            col_index = (index + 2) % 4

        plot_metric_vs_epoch(axes[row_index, col_index], KL_individual_storers, title, "epoch", "loss", y_scale='log', index=index)

    display_lowest_loss(axes[row_index, col_index+1], KL_individual_storers, "for whole model")

    for i in range(2):
        for j in range(4):
            current_ax = axes[n_rows - 2 + i, j]
            
            # Check if the content in the current axes is created by display_lowest_loss
            if not current_ax.lines and not current_ax.texts:
                fig.delaxes(current_ax)

#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________

# Plotter function to plot the results of training independent of the number of matrices used
def ALLplotting_5DMatrix_variable_number_of_matrices_4(canonical_beam, predicted_beam, KL_individual_storers, bandwidth, number_of_matrices, dataset_num, scenario_used, positions_, a, b, bins):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
        predicted_beam: training data for the beam
        KL_storer: List to store KL divergence values
        KL_storerA: List to store KL divergence values for matrix A
        KL_storerB: List to store KL divergence values for matrix B
        distance_storer: List to store the value of distance which we use to add a penalty if the simulated beam particles are outside the viewers dimensions
        dataset_num: The dataset number
    """
    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c

    # Helper function to set the range for 2D histograms
    def histogram_range(b):
        x_range = (-b, b)
        y_range = (-b, b)
        return x_range, y_range

    # Helper function to plot a KDE (Kernel Density Estimation) of data
    def KDE_intra_plotter(ax, beam, title, a = 150, b = 150):
        loaded = loader(beam[:,0],beam[:,2])
        KDE_plotter_ax(ax, kde(loaded), a = a, b = b)
        ax.set_title(title)


    # Helper function to plot a 2D histogram
    def plot_2d_histogram(ax, data, x_col, y_col, title, x_label, y_label, bins=30):
        x_range, y_range = histogram_range(margin_setter(data[:, x_col], data[:, y_col]) * 1.2)
        h = ax.hist2d(data[:, x_col].numpy(), data[:, y_col].numpy(), bins=bins, cmap='coolwarm', range=[x_range, y_range])
        ax.set_title(title)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        cbar = plt.colorbar(h[3], ax=ax)

    # Helper function to plot the residue (difference between two 2D histograms)
    def plot_residue(axes, x_col, y_col, data1, data2, title, x_label, y_label, bins):
        x_range, y_range = histogram_range(margin_setter(data1[:, x_col], data1[:, y_col]) * 1.2)
        hist1, xedges, yedges = np.histogram2d(data1[:, x_col].numpy(), data1[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        hist2, _, _ = np.histogram2d(data2[:, x_col].numpy(), data2[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        residue = hist1 - hist2
        residue_rotated = np.rot90(residue, k=1)
        ax = sns.heatmap(residue_rotated, cmap='coolwarm', annot=False, cbar=True, ax=axes)
        cbar = ax.collections[0].colorbar
        axes.set_title(title, fontsize=21)
        axes.set_xlabel(x_label, fontsize=22)
        axes.set_ylabel(y_label, fontsize=22)
        axes.set_xticks([])
        axes.set_yticks([])

    # Helper function to create a scatterplot
    def scatterplot_scatter(ax, x1, x2, y1, y2, label1, label2, title, x_label, y_label):
        ax.scatter(y1, y2, label=label2, marker="x", s=6)
        ax.scatter(x1, x2, label=label1, s=6)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        margin = margin_setter(y1, y2) * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.legend()

    # Helper function to plot a metric vs. epoch plot
    def plot_metric_vs_epoch(ax, data, title, x_label, y_label, y_scale='linear', index = None):
        if len(data) == 3 and index is not None:
            ax.plot(range(len(data[0][index])), data[0][index], label = "total loss", color = "green")
            ax.plot(range(len(data[1][index])), data[1][index], label = "kl", color = "blue")
            ax.plot(range(len(data[2][index])), data[2][index], label = "distance", color = "red")
            ax.set_title(title)
            ax.set_yscale(y_scale)
            ax.set_xlabel(x_label, fontsize=22)
            ax.set_ylabel(y_label, fontsize=22)
            ax.legend()
        else:
            ax.plot(range(len(data)), data)
            ax.set_title(title)
            ax.set_yscale(y_scale)
            ax.set_xlabel(x_label, fontsize=22)
            ax.set_ylabel(y_label, fontsize=22)


    #This function prints the lowest KL for a given matrix
    def display_lowest_loss(ax, loss_storer, beam_origin, index  = None):
        if index is None:
            index = 0
            
        if len(loss_storer) == 3:
            ax.text(0.5, 0.5, f"Lowest loss for {beam_origin}: {min(loss_storer[0][index]):.3e}", transform=ax.transAxes, fontsize=25, ha='center', fontweight="bold", va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')    
        else:
            ax.text(0.5, 0.5, f"Lowest loss for {beam_origin}: {min(loss_storer):.3e}", transform=ax.transAxes, fontsize=25, ha='center', fontweight="bold", va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')  
              

    def find_number_of_rows(number):
        for index in range(number):
            # Calculate row and column indices dynamically
            row_index = 5 + (index - 1) // 2
            col_index = 2 * ((index - 1) % 2)

        last_col_index = col_index
        last_row_index = row_index

        for index in range (1 + number):
            # Calculate row and column indices dynamically
            if (last_col_index + 1) == 3:
                row_index = last_row_index + (index) // 4
                row_index = row_index + 1 
                col_index = (index) % 4
            else:
                row_index = last_row_index + (index + 2) // 4
                col_index = (index + 2) % 4

        return row_index+1
        
    # Initialize bandwidth and positions for Kernel Density Estimation (KDE)
    kde = KDEGaussian(bandwidth=bandwidth, locations=positions_)
    
    # Create subplots for different plots
    n_rows = find_number_of_rows(number_of_matrices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(46, 8*n_rows))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    mpl.rcParams['font.size'] = 25

    beam1 = predicted_beam
    beam2 = canonical_beam
    beam1_name = "predicted_beam"
    beam2_name = "canonical_beam"

    with open(scenario_used, "r") as f:
        lines = f.readlines()

    scatterplot_scatter(axes[0], beam1[:, 0], beam1[:, 2], beam2[:, 0], beam2[:, 2], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "y")
    scatterplot_scatter(axes[1], beam1[:, 0], beam1[:, 1], beam2[:, 0], beam2[:, 1], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "aX")
    scatterplot_scatter(axes[2], beam1[:, 2], beam1[:, 3], beam2[:, 2], beam2[:, 3], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "y", "aY")
    scatterplot_scatter(axes[3], beam1[:, 0], beam1[:, 5], beam2[:, 0], beam2[:, 5], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "dE")

    plot_2d_histogram(axes[4], beam2, 0, 2, f"{beam2_name} y vs x", "x", "y", bins)
    plot_2d_histogram(axes[5], beam2, 0, 1, f"{beam2_name} aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[6], beam2, 2, 3, f"{beam2_name} aY vs y", "y", "aY", bins)
    plot_2d_histogram(axes[7], beam2, 0, 5, f"{beam2_name} dE vs X", "dE", "X", bins)

    plot_2d_histogram(axes[8], beam1, 0, 2, f"{beam1_name} y vs x", "x", "y", bins)
    plot_2d_histogram(axes[9], beam1, 0, 1, f"{beam1_name} aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[10], beam1, 2, 3, f"{beam1_name} aY vs y", "y", "aY", bins)
    plot_2d_histogram(axes[11], beam1, 0, 5, f"{beam1_name} dE vs X", "dE", "X", bins)

    
    plot_residue(axes[12], 0, 2, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'Y', bins)
    plot_residue(axes[13], 0, 1, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'aX', bins)
    plot_residue(axes[14], 2, 3, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'Y', 'aY', bins)
    plot_residue(axes[15], 0, 5, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'dE', bins)

    KDE_intra_plotter(axes[16], beam1, f"dataset {dataset_num} {beam1_name}", a = a, b = b)
    KDE_intra_plotter(axes[17], beam2, f"dataset {dataset_num} {beam2_name}", a = a, b = b)


    for index in range(number_of_matrices):
        coeff_array, power_array = read_map(lines[index][:-1])   

        beam1= transport(predicted_beam, coeff_array, power_array)
        beam2 = transport(canonical_beam, coeff_array, power_array)
        beam_origin = lines[index][7:-5]

        index_pred = 18 + (index*2) 
        index_real = 18 + (index*2) + 1
        KDE_intra_plotter(axes[index_pred], beam2, f"Pred beam for {beam_origin}", a = a, b = b)
        KDE_intra_plotter(axes[index_real], beam1, f"Canon beam for {beam_origin}", a = a, b = b)

    for index in range(number_of_matrices):
        # create plot name dynamicaly
        if index == 0:
            title = "Total loss vs epoch"
        else:
            beam_origin = lines[index][7:-5]
            title = f"loss for Matrix {beam_origin}"

        index_loss = index_real + 1 + index

        plot_metric_vs_epoch(axes[index_loss], KL_individual_storers, title, "epoch", "loss", y_scale='log', index=index)

    index_loss = index_loss + 1 

    display_lowest_loss(axes[index_loss], KL_individual_storers[0], "for whole model")


#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________


# Helper function to calculate the margin
def ALLplotting_5DMatrix_variable_number_of_matrices_ABCDGHI(canonical_beam, predicted_beam, KL_individual_storers, bandwidth, number_of_matrices, dataset_num, bins):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
        predicted_beam: training data for the beam
        KL_storer: List to store KL divergence values
        KL_storerA: List to store KL divergence values for matrix A
        KL_storerB: List to store KL divergence values for matrix B
        distance_storer: List to store the value of distance which we use to add a penalty if the simulated beam particles are outside the viewers dimensions
        dataset_num: The dataset number
    """
    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c

    # Helper function to set the range for 2D histograms
    def histogram_range(b):
        x_range = (-b, b)
        y_range = (-b, b)
        return x_range, y_range

    # Helper function to plot a KDE (Kernel Density Estimation) of data
    def KDE_intra_plotter(ax, beam, title):
        loaded = loader(beam[:,0],beam[:,2])
        KDE_plotter_ax(ax, kde(loaded))
        ax.set_title(title)

    # Helper function to plot a 2D histogram
    def plot_2d_histogram(ax, data, x_col, y_col, title, x_label, y_label, bins=30):
        x_range, y_range = histogram_range(margin_setter(data[:, x_col], data[:, y_col]) * 1.2)
        h = ax.hist2d(data[:, x_col].numpy(), data[:, y_col].numpy(), bins=bins, cmap='coolwarm', range=[x_range, y_range])
        ax.set_title(title)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        cbar = plt.colorbar(h[3], ax=ax)

    # Helper function to plot the residue (difference between two 2D histograms)
    def plot_residue(axes, x_col, y_col, data1, data2, title, x_label, y_label, bins):
        x_range, y_range = histogram_range(margin_setter(data1[:, x_col], data1[:, y_col]) * 1.2)
        hist1, xedges, yedges = np.histogram2d(data1[:, x_col].numpy(), data1[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        hist2, _, _ = np.histogram2d(data2[:, x_col].numpy(), data2[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        residue = hist1 - hist2
        residue_rotated = np.rot90(residue, k=1)
        ax = sns.heatmap(residue_rotated, cmap='coolwarm', annot=False, cbar=True, ax=axes)
        cbar = ax.collections[0].colorbar
        axes.set_title(title, fontsize=21)
        axes.set_xlabel(x_label, fontsize=22)
        axes.set_ylabel(y_label, fontsize=22)
        axes.set_xticks([])
        axes.set_yticks([])

    # Helper function to create a scatterplot
    def scatterplot_scatter(ax, x1, x2, y1, y2, label1, label2, title, x_label, y_label):
        ax.scatter(x1, x2, label=label1, s=6)
        ax.scatter(y1, y2, label=label2, marker="x", s=6)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        margin = margin_setter(y1, y2) * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.legend()

    # Helper function to plot a metric vs. epoch plot
    def plot_metric_vs_epoch(ax, data, title, x_label, y_label, y_scale='linear'):
        ax.plot(range(len(data)), data)
        ax.set_title(title)
        ax.set_yscale(y_scale)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)

    #This function prints the lowest KL for a given matrix
    def display_lowest_loss(ax, loss_storer, beam_origin):
        ax.text(0.5, 0.5, f"Lowest loss for {beam_origin}: {min(loss_storer):.3e}", transform=ax.transAxes, fontsize=25, ha='center', fontweight="bold", va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')    

    def find_number_of_rows(number):
        for index in range(number):
            # Calculate row and column indices dynamically
            row_index = 5 + (index - 1) // 2
            col_index = 2 * ((index - 1) % 2)

        last_col_index = col_index
        last_row_index = row_index

        for index in range (1 + number):
            # Calculate row and column indices dynamically
            if (last_col_index + 1) == 3:
                row_index = last_row_index + (index) // 4
                row_index = row_index + 1 
                col_index = (index) % 4
            else:
                row_index = last_row_index + (index + 2) // 4
                col_index = (index + 2) % 4

        return row_index+1
        
    # Initialize bandwidth and positions for Kernel Density Estimation (KDE)
    positions_ = bandloc(set_it=True) #<----- This is the values we have been using for reference (set_it is set to True)
    kde = KDEGaussian(bandwidth=bandwidth, locations=positions_)
    
    # Create subplots for different plots
    n_rows = find_number_of_rows(number_of_matrices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(46, 8*n_rows))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    mpl.rcParams['font.size'] = 25
       

    beam1 = predicted_beam
    beam2 = canonical_beam
    beam1_name = "predicted_beam"
    beam2_name = "canonical_beam"

    scatterplot_scatter(axes[0], beam1[:, 0], beam1[:, 2], beam2[:, 0], beam2[:, 2], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "y")
    scatterplot_scatter(axes[1], beam1[:, 0], beam1[:, 1], beam2[:, 0], beam2[:, 1], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "aX")
    scatterplot_scatter(axes[2], beam1[:, 2], beam1[:, 3], beam2[:, 2], beam2[:, 3], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "y", "aY")
    scatterplot_scatter(axes[3], beam1[:, 0], beam1[:, 5], beam2[:, 0], beam2[:, 5], beam1_name, beam2_name, f"{beam1_name} vs {beam2_name}", "x", "dE")

    plot_2d_histogram(axes[4], beam2, 0, 2, f"{beam2_name} y vs x", "x", "y", bins)
    plot_2d_histogram(axes[5], beam2, 0, 1, f"{beam2_name} aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[6], beam2, 2, 3, f"{beam2_name} aY vs y", "y", "aY", bins)
    plot_2d_histogram(axes[7], beam2, 0, 5, f"{beam2_name} dE vs X", "dE", "X", bins)

    plot_2d_histogram(axes[8], beam1, 0, 2, f"{beam1_name} y vs x", "x", "y", bins)
    plot_2d_histogram(axes[9], beam1, 0, 1, f"{beam1_name} aX vs x", "x", "aX", bins)
    plot_2d_histogram(axes[10], beam1, 2, 3, f"{beam1_name} aY vs y", "y", "aY", bins)
    plot_2d_histogram(axes[11], beam1, 0, 5, f"{beam1_name} dE vs X", "dE", "X", bins)

    
    plot_residue(axes[12], 0, 2, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'Y', bins)
    plot_residue(axes[13], 0, 1, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'aX', bins)
    plot_residue(axes[14], 2, 3, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'Y', 'aY', bins)
    plot_residue(axes[15], 0, 5, beam1, beam2, f'Residue ({beam1_name} - {beam2_name})', 'X', 'dE', bins)

    KDE_intra_plotter(axes[16], beam1, f"dataset {dataset_num} {beam1_name}")
    KDE_intra_plotter(axes[17], beam2, f"dataset {dataset_num} {beam2_name}")

    for index in range(number_of_matrices):
        # Current matrix letter
        current_letter = "ABCDGHI"[index]

        # Transport data using coefficient arrays and power arrays
        matrix_filename = f'matrix {current_letter}.txt'
        coeff_array, power_array = read_map(f'./matrices/{matrix_filename}')  
        beam1= transport(predicted_beam, coeff_array, power_array)
        beam2 = transport(canonical_beam, coeff_array, power_array)
        beam_origin = f'matrix {current_letter}'

        index_pred = 19 + (index*2) 
        index_real = 19 + (index*2) + 1
        KDE_intra_plotter(axes[index_pred], beam2, f"dataset {dataset_num} {beam_origin} {beam2_name}")
        KDE_intra_plotter(axes[index_real], beam1, f"dataset {dataset_num} {beam_origin} {beam1_name}")

    for index in range(len(KL_individual_storers)):
        # create plot name dynamicaly
        if index == 0:
            title = "Total loss vs epoch"
        else:
            current_letter = "ABCDGHI"[index-1]
            title = f"loss for matrix {current_letter} vs epoch"

        index_loss = index_real + 1 + index

        plot_metric_vs_epoch(axes[index_loss], KL_individual_storers[index], title, "epoch", "KL", y_scale='log')

    display_lowest_loss(axes[index_loss+1], KL_individual_storers[0], "for whole model")

#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________

# BIG graph that has all the information one may need for a 3D matrix (MARK IV).
def ALLplotting_4DMatrix_I(canonical_beam, predicted_beam, KL_storer, KL_storerA, KL_storerB, distance_storer, dataset_num):
    """
    Args:
        canonical_beam: the initial beam we are looking for.
        predicted_beam: training data for the beam
        KL_storer: List to store KL divergence values
        KL_storerA: List to store KL divergence values for matrix A
        KL_storerB: List to store KL divergence values for matrix B
        distance_storer: List to store the value of distance which we use to add a penalty if the simulated beam particles are outside the viewers dimensions
        dataset_num: The dataset number
    """

    # Helper function to calculate the margin
    def margin_setter(datax, datay):
        a = torch.max(torch.abs(datax)).item()
        b = torch.max(torch.abs(datay)).item()
        c = np.max([a,b])
        return c
    
    # Helper function to set the range for 2D histograms
    def histogram_range(b):
        x_range = (-b, b)
        y_range = (-b, b)
        return x_range, y_range
    
    # Helper function to plot a KDE (Kernel Density Estimation) of data
    def KDE_intra_plotter(ax, rayo, matrix_number, title):
        coeff_array, power_array = read_map(f'matrix{matrix_number}.txt' ) # 'matrix1.txt'  
        beam = transport(rayo, coeff_array, power_array)
        loaded = loader(beam[:,0],beam[:,2])
        KDE_plotter_ax(ax, kde(loaded))
        ax.set_title(title)

    # Helper function to plot a 2D histogram
    def plot_2d_histogram(ax, data, x_col, y_col, title, x_label, y_label, bins=30):
        x_range, y_range = histogram_range(margin_setter(data[:, x_col], data[:, y_col]) * 1.2)
        h = ax.hist2d(data[:, x_col].numpy(), data[:, y_col].numpy(), bins=bins, cmap='coolwarm', range=[x_range, y_range])
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        cbar = plt.colorbar(h[3], ax=ax)

    # Helper function to plot the residue (difference between two 2D histograms)
    def plot_residue(axes, x_col, y_col, data1, data2, title, x_label, y_label):
        x_range, y_range = histogram_range(margin_setter(data1[:, x_col], data1[:, y_col]) * 1.2)
        hist1, xedges, yedges = np.histogram2d(data1[:, x_col].numpy(), data1[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        hist2, _, _ = np.histogram2d(data2[:, x_col].numpy(), data2[:, y_col].numpy(), bins=bins, range=[x_range, y_range])
        residue = hist1 - hist2
        residue_rotated = np.rot90(residue, k=1)
        ax = sns.heatmap(residue_rotated, cmap='coolwarm', annot=False, cbar=True, ax=axes)
        cbar = ax.collections[0].colorbar
        axes.set_title(title)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.set_xticks([])
        axes.set_yticks([])

    # Helper function to create a scatterplot
    def scatterplot_scatter(ax, x1, x2, y1, y2, label1, label2, title, x_label, y_label):
        ax.scatter(x1, x2, label=label1, s=6)
        ax.scatter(y1, y2, label=label2, marker="x", s=6)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        margin = margin_setter(y1, y2) * 1.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.legend()

    # Helper function to plot a metric vs. epoch plot
    def plot_metric_vs_epoch(ax, data, title, x_label, y_label, y_scale='linear'):
        ax.plot(range(len(data)), data)
        ax.set_title(title)
        ax.set_yscale(y_scale)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
    # Initialize bandwidth and positions for Kernel Density Estimation (KDE)
    bandwidth, positions_ = bandloc(set_it=True) #<----- This is the values we have been using for reference (set_it is set to True)
    kde = KDEGaussian(bandwidth=0.005, locations=positions_)
    
    # Create subplots for different plots
    fig, axes = plt.subplots(12, 4, figsize=(20, 40))
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)
    mpl.rcParams['font.size'] = 9
    
    # Plot KDE for matrix A and matrix B density predictions and real values
    KDE_intra_plotter(axes[0,0], predicted_beam, 1, f"dataset {dataset_num} matrix A predicted_density")
    KDE_intra_plotter(axes[0,1], canonical_beam, 1, f"dataset {dataset_num} matrix A canonical_density")
    KDE_intra_plotter(axes[0,2], predicted_beam, 2, f"dataset {dataset_num} matrix B predicted_density")
    KDE_intra_plotter(axes[0,3], canonical_beam, 2, f"dataset {dataset_num} matrix B canonical_density")
        
    bins = 30

    # Plot 2D histograms for various data
    plot_2d_histogram(axes[1, 0], canonical_beam, 0, 2, "canonical_beam Y vs X", "X", "Y", bins)
    plot_2d_histogram(axes[1, 1], canonical_beam, 0, 1, "canonical_beam aX vs X", "X", "aX", bins)
    plot_2d_histogram(axes[1, 2], canonical_beam, 2, 3, "canonical_beam Y vs aY", "Y", "aY", bins)
    plot_2d_histogram(axes[1, 3], predicted_beam, 0, 2, "predicted_beam Y vs X", "X", "Y", bins)
    plot_2d_histogram(axes[2, 0], predicted_beam, 0, 1, "predicted_beam aX vs X", "X", "aX", bins)
    plot_2d_histogram(axes[2, 1], predicted_beam, 2, 3, "predicted_beam Y vs aY", "Y", "aY", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[2, 2], 0, 2, canonical_beam, predicted_beam, 'Residue (canonical_beam - predicted_beam)', 'X', 'Y')
    plot_residue(axes[2, 3], 0, 1, canonical_beam, predicted_beam, 'Residue (canonical_beam - predicted_beam)', 'X', 'aX')
    plot_residue(axes[3, 0], 2, 3, canonical_beam, predicted_beam, 'Residue (canonical_beam - predicted_beam)', 'Y', 'aY')

    # Create scatterplots to compare data
    scatterplot_scatter(axes[3, 1], canonical_beam[:, 0], canonical_beam[:, 2], predicted_beam[:, 0], predicted_beam[:, 2], "canonical_beam", "predicted_beam", "canonical_beam vs predicted_beam", "X", "Y")
    scatterplot_scatter(axes[3, 2], canonical_beam[:, 0], canonical_beam[:, 1], predicted_beam[:, 0], predicted_beam[:, 1], "canonical_beam", "predicted_beam", "canonical_beam vs predicted_beam", "X", "aX")
    scatterplot_scatter(axes[3, 3], canonical_beam[:, 2], canonical_beam[:, 3], predicted_beam[:, 2], predicted_beam[:, 3], "canonical_beam", "predicted_beam", "canonical_beam vs predicted_beam", "Y", "aY")

    # Transport data using coefficient arrays and power arrays
    coeff_array, power_array = read_map('matrix1.txt')  
    postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

    # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data
    plot_2d_histogram(axes[4, 0], postCOSY_pred_beam, 0, 2, "postCOSY_pred_beam matrix 1 Y vs X", "X", "Y", bins)
    plot_2d_histogram(axes[4, 1], postCOSY_pred_beam, 0, 1, "postCOSY_pred_beam matrix 1 aX vs X", "X", "aX", bins)
    plot_2d_histogram(axes[4, 2], postCOSY_pred_beam, 2, 3, "postCOSY_pred_beam matrix 1 aY vs Y", "Y", "aY", bins)
    plot_2d_histogram(axes[4, 3], postCOSY_canon_beam, 0, 2, "postCOSY_canon_beam matrix 1 Y vs X", "X", "Y", bins)
    plot_2d_histogram(axes[5, 0], postCOSY_canon_beam, 0, 1, "postCOSY_canon_beam matrix 1 aX vs X", "X", "aX", bins)
    plot_2d_histogram(axes[5, 1], postCOSY_canon_beam, 2, 3, "postCOSY_canon_beam matrix 1 aY vs Y", "Y", "aY", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[5, 2], 0, 2, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix A (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'Y')
    plot_residue(axes[5, 3], 0, 1, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix A (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'aX')
    plot_residue(axes[6, 0], 2, 3, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix A (postCOSY_canon_beam - postCOSY_pred_beam)', 'Y', 'aY')
    
    # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix A
    scatterplot_scatter(axes[6, 1], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 2], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "X", "Y")
    scatterplot_scatter(axes[6, 2], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 1], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 1], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "X", "aX")
    scatterplot_scatter(axes[6, 3], postCOSY_canon_beam[:, 2], postCOSY_canon_beam[:, 3], postCOSY_pred_beam[:, 2], postCOSY_pred_beam[:, 3], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix A", "Y", "aY")

    # Transport data using coefficient arrays and power arrays for matrix B
    coeff_array, power_array = read_map('matrix2.txt')  
    postCOSY_pred_beam = transport(predicted_beam, coeff_array, power_array)
    postCOSY_canon_beam = transport(canonical_beam, coeff_array, power_array)

    # Plot 2D histograms for postCOSY_pred_beam and postCOSY_canon_beam data for matrix B
    plot_2d_histogram(axes[7, 0], postCOSY_pred_beam, 0, 2, "postCOSY_pred_beam matrix 2 y vs x", "X", "Y", bins)
    plot_2d_histogram(axes[7, 1], postCOSY_pred_beam, 0, 1, "postCOSY_pred_beam matrix 2 aX vs x", "X", "aX", bins)
    plot_2d_histogram(axes[7, 2], postCOSY_pred_beam, 2, 3, "postCOSY_pred_beam matrix 2 aX vs x", "Y", "aY", bins)
    plot_2d_histogram(axes[7, 3], postCOSY_canon_beam, 0, 2, "postCOSY_canon_beam matrix 2 y vs x", "X", "Y", bins)
    plot_2d_histogram(axes[8, 0], postCOSY_canon_beam, 0, 1, "postCOSY_canon_beam matrix 2 aX vs x", "X", "aY", bins)
    plot_2d_histogram(axes[8, 1], postCOSY_canon_beam, 2, 3, "postCOSY_canon_beam matrix 2 aX vs x", "Y", "aY", bins)

    # Plot residue (difference) between canonical_beam and predicted_beam
    plot_residue(axes[8, 2], 0, 2, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix B (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'Y')
    plot_residue(axes[8, 3], 0, 1, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix B (postCOSY_canon_beam - postCOSY_pred_beam)', 'X', 'aX')
    plot_residue(axes[9, 0], 2, 3, postCOSY_canon_beam, postCOSY_pred_beam, 'Residue matrix B (postCOSY_canon_beam - postCOSY_pred_beam)', 'Y', 'aY')

    # Create scatterplots to compare postCOSY_pred_beam and postCOSY_canon_beam data for matrix B
    scatterplot_scatter(axes[9, 1], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 2], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 2], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix B", "X", "Y")
    scatterplot_scatter(axes[9, 2], postCOSY_canon_beam[:, 0], postCOSY_canon_beam[:, 1], postCOSY_pred_beam[:, 0], postCOSY_pred_beam[:, 1], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix B", "X", "aX")
    scatterplot_scatter(axes[9, 3], postCOSY_canon_beam[:, 2], postCOSY_canon_beam[:, 3], postCOSY_pred_beam[:, 2], postCOSY_pred_beam[:, 3], "postCOSY_canon_beam", "postCOSY_pred_beam", "postCOSY_pred_beam vs postCOSY_canon_beam - matrix B", "y", "ay")

    # Plot KL divergence, distance, and their relationships
    plot_metric_vs_epoch(axes[10, 0], KL_storer, "Total KL vs epoch", "epoch", "KL", y_scale='log')
    plot_metric_vs_epoch(axes[10, 1], KL_storerA, "KL matrix A vs epoch", "epoch", "KL", y_scale='log')
    plot_metric_vs_epoch(axes[10, 2], KL_storerB, "KL matrix B vs epoch", "epoch", "KL", y_scale='log')
    plot_metric_vs_epoch(axes[10, 3], distance_storer, "distance vs epoch", "epoch", "margin distance", y_scale='log')

    # Scatterplot of KL matrix B vs KL matrix A
    axes[11,0].scatter(KL_storerA, KL_storerB)
    axes[11,0].set_title("KL matrix B vs KL matrix A")
    axes[11,0].set_xscale('log')
    axes[11,0].set_yscale('log')
    axes[11,0].set_xlabel("KL_A")
    axes[11,0].set_ylabel("KL_B")

    # Scatterplot of Total KL vs distance
    axes[11,1].scatter(distance_storer[:len(KL_storer)], KL_storer)
    axes[11,1].set_title("Total KL vs distance")
    axes[11,1].set_ylabel("total KL")
    axes[11,1].set_yscale('log')
    axes[11,1].set_xlabel("distance")
    
    axes[11,2].text(0.5, 0.5, f"Final KL Train Loss: {min(KL_storer):.3e}", transform=axes[11, 2].transAxes, fontsize = 15, ha='center', fontweight="bold", va='center')
    axes[11,2].set_xlim(0, 1)
    axes[11,2].set_ylim(0, 1)
    axes[11,2].axis('off')

    axes[11, 3].text(0.5, 0.2, f"FinalLoss for matrix A: {min(KL_storerA):.3e}", transform=axes[11, 3].transAxes, fontsize=15, ha='center', fontweight="bold", va='center')
    axes[11, 3].text(0.5, 0.8, f"Final loss for matrix B: {min(KL_storerB):.3e}", transform=axes[11, 3].transAxes, fontsize=15, ha='center', fontweight="bold", va='center')
    axes[11, 3].set_xlim(0, 1)
    axes[11, 3].set_ylim(0, 1)
    axes[11, 3].axis('off')

    plt.tight_layout()



