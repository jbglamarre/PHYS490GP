#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:22:59 2021
Author: Matthew Smith
Student ID: 20663244
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk #Basic GUI
import os
from tkinter import filedialog

#-------------------------------AskForFile------------------------------------


def AskForPath(v): 
    '''
    prompts user to identify a file path containing data 
    
    input: v int - verbosity
    output: the selected directory
    '''
    
    root = tk.Tk() # make tk object
    root.withdraw()
    fpath = filedialog.askdirectory() 
    root.update()
    root.destroy()

    if v>=1:
        print('Importing from: ' + fpath)
        
    assert fpath!= '', 'Please select valid directory' # if use cancles
    
    return fpath


#----------------------------Get_File_Paths------------------------------------


def Get_File_Paths():
    '''
    
    Simple helper function to ask user
    to identify the directory  of the data

    Returned values:
        - file_Paths: list of all relevent file's paths
        
    '''
    file_Paths = []
    path = AskForPath(1)    # asks for path
    
    # make list of allrelevent files in that path
    for filename in os.listdir(path): 
        print(filename)
        if filename.endswith(".npz"):
            print(filename)
            file_Paths.append(os.path.join(path, filename))
            
    return file_Paths


#-------------------------------Data_Loader------------------------------------


def Data_Loader( file_Path, files_per_batch ):
    '''
    This function trains the network.

    Arguments:

        - file_Path: Str containing the path to a single data file 
        - files_per_batch: int indicating how many files to train on at one time 

    Returned values:

            images : [ batchsize , 1 , 128 , 128 ]
            lables : [batchsize , 1]
    '''
    # import data
    data = np.load(file_Path,allow_pickle=True) 
    
    images = np.array(data['images'])
    labels = np.array(data['labels'])
    
    return images, labels

#-------------------------------Data_Loader------------------------------------

def potential_plot_gen( plot , num_pot , x , y , V_batch ):
    '''
     This function generates figures for any potential type with 3D and 2D subplots.
      
     Arguments:
      
    		- self: Class instance.
    		- plot: Whether to plot (True) or not (False) (dtype = bool).
    '''
      
    if plot == True:
      
    	# Plots every potentials if num_pot <= maxplotlib's maximum plot limit of 20.
    	if num_pot <= 20:
    		for ipot in range( 0 , num_pot ):
    			fig = plt.figure( )
      
    			ax1 = fig.add_subplot( 1 , 2 , 1 , projection = "3d" )
    			ax1.plot_surface( x , y , V_batch[ ipot , : , : ] )
      
    			ax2 = fig.add_subplot( 1 , 2 , 2 )
    			ax2.imshow( V_batch[ ipot , : , : ] , origin = "lower" , cmap = "magma_r" )
      
    	# Adjusts to only plot the first 20 potentials if num_pot > maxplotlib's maximum plot limit of 20.
    	elif num_pot > 20:
    		for ipot in range( 0 , 21 ):
    			fig = plt.figure( )
      
    			ax1 = fig.add_subplot( 1 , 2 , 1 , projection = "3d" )
    			ax1.plot_surface( x , y , V_batch[ ipot , : , : ] )
      
    			ax2 = fig.add_subplot( 1 , 2 , 2 )
    			ax2.imshow( V_batch[ ipot , : , : ] , origin = "lower" , cmap = "magma_r" )
      
    	plt.show( )
    	plt.close( )
        
   
    
def main():
    
    image_size = 128
    
    # get path
    file_Path = Get_File_Paths()

    for path in file_Path:
        
            # load the images and lables for the specified files
            images, labels = Data_Loader( path, 1 )
            # creat meshgrid for plotting
            x,y = np.meshgrid( np.linspace( - 20 , 20 , image_size ) , np.linspace( - 20 , 20 , image_size ) )
            
            # plot potentials
            potential_plot_gen( True , 1000 , x , y , images )
            
main()
 
