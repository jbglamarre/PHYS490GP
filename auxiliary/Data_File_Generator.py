#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:07:01 2021
author: Matthew Smith
Student ID: 20663244
"""

"""
For usage, see main Function
"""
 
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import os, time 
# class createdby mike - This code needs to be in working directory
from data_gen import potentials_gen 
from solver import solver
#Basic GUI
import tkinter as tk 
import tkinter.filedialog
# Progress Bar
from tqdm.auto import tqdm
# REGEX for string matching
import re
import argparse

# =============================================================================
# Helper Functions
# =============================================================================


#-------------------------------AskForFile------------------------------------

def AskForPath(v): 
    '''
        Prompts user to open a file path to be used 
        
    Arguments: 
        - v: verbosity (dtype = int)
    Returned Values: 
        - fpath: the selected directory
    '''
    root = tk.Tk() # make tk object
    root.withdraw()
    fpath= tk.filedialog.askdirectory() 
    root.update()
    #fname = askopenfilename(title="Select Folder to save to ", filetypes=(("text files","*.txt"),("all files","*.*")))
    if v>=1:
        print('Exporting to: ' + fpath)
        
    assert fpath!= '', 'Please select valid directory' # if use cancles
    
    return fpath


#-------------------------------Sorter------------------------------------
  
def Sorter(image_size, num_pot_sho, num_pot_iw, num_pot_dig, num_pot_rnd, plot ,v, sol):
    '''
        Generates the number of potentials as specified in entries   
        
    Arguments: 
        - image_size: (dtype = int)
        - num_pot_...: (dtype = int)
        - plot: Should stay as False (dtype = Boolean)
        - v: verbosity (dtype = int)
        - sol: Class object from the solver class

    Returned Values:
        - out: ordered array with the potentials (dtype = array)
        - label_list: ordered list of lables (dtype = array)
        - fineName: str with file names (dtype = str)
    '''
    
    
    num_pot = num_pot_sho + num_pot_iw + num_pot_dig + num_pot_rnd
    assert (num_pot)  > 0 , "number of total potentials needs to be greater then 0"
    
    
    #pot_dict= {}
    pot_list = [] # adds all potentials inot a list for conditional concatination at bottom
    label_list = [0]*num_pot
    fileName = '_'
    
    
    #------------------------------------SHO-----------------------------------
    # specific formating on SHO to test and validate potentials if need be
 
    if num_pot_sho!=0:# if statements to generate the amount of data asked. If amount of 1 potential = 0, pass it
        
        fileName += 'SHO' + str(num_pot_sho) + '_'
        pot_sho_object = potentials_gen( image_size , num_pot_sho ).sho_potential_gen(plot) 
        pot_sho = pot_sho_object.V_batch # get potential stack
        pot_list.append(pot_sho) # add to list of all potentials

        for count,pot in enumerate(pot_sho):
            label_list[count] = sol.solve(pot)  
                
        if v >=3:
            print( str(num_pot_sho) +'_SHO_' + 'Generated')
            if v>3: # test ot make sure solution is correct
                for count in range(pot_sho.shape[0]):
                    E_SHO_analytic = 0.5 * ( np.sqrt(pot_sho_object.k_x[count]) + np.sqrt(pot_sho_object.k_y[count]) )
                    print('comparing analytic & solver: ' + str(E_SHO_analytic) + ',' + str(label_list[count]))
                    print('percent difference: ' + str(((E_SHO_analytic - label_list[count]) /E_SHO_analytic) * 100 ))
            
    #------------------------------------IW-----------------------------------                       
    if num_pot_iw!=0:
        
        fileName += 'IW'  + str(num_pot_iw) + '_'
        pot_iw = potentials_gen( image_size , num_pot_iw ).iw_potential_gen(plot).V_batch 
        pot_list.append(pot_iw)
        
        for count,pot in enumerate(pot_iw):
            label_list[count + num_pot_sho ] = sol.solve(pot)    
        '''    
        if v >=3:
            print( str(num_pot_iw) +'_IW_' + 'Generated')
        '''
    #------------------------------------DIG----------------------------------       
    if num_pot_dig!=0:
        
        fileName += 'DIG'  + str(num_pot_dig) + '_'
        pot_dig = potentials_gen( image_size , num_pot_dig ).dig_potential_gen(plot).V_batch
        pot_list.append(pot_dig)
        
        for count,pot in enumerate(pot_dig):
            label_list[count + num_pot_sho + num_pot_iw ] = sol.solve(pot)  
        '''    
        if v >=3:
            print( str(num_pot_dig) +'_DIG_' + 'Generated')
        ''' 
    #------------------------------------RND----------------------------------           
    if num_pot_rnd!=0:
        
        fileName += 'RND'  + str(num_pot_rnd) + '_'
        pot_rnd = potentials_gen( image_size , num_pot_rnd ).rnd_potential_gen(plot).V_batch
        pot_list.append(pot_rnd)
        
        for count,pot in enumerate(pot_rnd):
            label_list[count + num_pot_sho + num_pot_iw + num_pot_dig] = sol.solve(pot) 
        '''
        if v >=3:
            print( str(num_pot_rnd) +'_RND_' + 'Generated')
        '''
   
   #-------------------------Final array concatinations-----------------------
    # conditionals so that you can take any combo of potentials 
    if len(pot_list) == 1:
        out = pot_list[0]
    elif len(pot_list) == 2:
        out = np.concatenate((pot_list[0], pot_list[1]),axis=0 )
    elif len(pot_list) == 3:
        out = np.concatenate((pot_list[0], pot_list[1], pot_list[2]),axis=0 )
    else:
        out = np.concatenate((pot_list[0], pot_list[1], pot_list[2], pot_list[3]),axis=0 )
        

    label_list = np.reshape(np.array(label_list), (len(label_list),1))

    return out, label_list, fileName #, pot_dict


# =============================================================================
# main
# =============================================================================
def main():
    
    """
    Usage:
    -Inside main function, simply input the number of each type of potentials you want in the output file
    -"num_files" Change number of files you want to make by changing 
    -Will prompt user and ask for the directory to save the files too
    -Output files will have naming "fileNameStart_ImageSize_itterater_potType #potential"
    
         -Ex: "Train_256_SHO250_IW250_DIG250_RND250_2.npy"
             -"Train" was the filename Start
             - image size of 256x256 
             - 250 SHO images in file
             - 250 IW images in file
             ...
             - 2nd file in a batch of x files made
    """
    parser = argparse.ArgumentParser(description='Data File Generation Parameters')

    parser.add_argument('--num_files', required=True, type=int, help='Number of Files to be written' )
    parser.add_argument('--image_size', required=False, type=int, default=128,  help='The image size of the potentials' )   
    #parser.add_argument('--potential', required=True, type=int, nargs=4 , choices=['sho', 'iw', 'dig', rnd]  help='Set the type of potentials' )
    parser.add_argument('--num_sho', required=False, default=0, type=int, help='Number of SHO potentials to be written' )
    parser.add_argument('--num_iw', required=False, default=0, type=int, help='Number of IW potentials to be written' )
    parser.add_argument('--num_dig', required=False, default=0, type=int, help='Number of DIG potentials to be written' )
    parser.add_argument('--num_rnd', required=False, default=0, type=int, help='Number of SHO potentials to be written' )
    parser.add_argument('--dataset_type', required=True, type=str, help='Create Training or Test Dataset' )
    parser.add_argument('--verbosity', required=False, default=3, type=int, help='Verbosity ( Max & Default: 3)' )
    args = parser.parse_args()
    
 ##############################################################################
    #                    Parameters to be altered

    num_files       = args.num_files        # Number of files to be written
    image_size      = args.image_size       # MUST BE 256 CURRENTLY - image dimensions
    num_pot_sho     = args.num_sho          # Number of SHO potentials you want in the file
    num_pot_iw      = args.num_iw           # Number of infinate well
    num_pot_dig     = args.num_dig          # Number of double inverted gaussian
    num_pot_rnd     = args.num_rnd          # Number of random potentials you want in the file
    v               = args.verbosity        # verbosity

    fileNameStart   = args.dataset_type 
    
    ##############################################################################
    
    
    # Change filename to Train or Test using regex
    if re.search( fileNameStart , "Train", re.IGNORECASE):
        fileNameStart = "Train_"

    elif re.search( fileNameStart , "Test", re.IGNORECASE) :
        fileNameStart = "Test_"


    start_time = time.time() # stars runtime counter
    path = AskForPath(v)    # asks for path 
    solve = solver( grid_size = image_size )

    # Creates progress bar
    pbar = tqdm( range(1, num_files+1), desc= "Creating files...", leave=True, position=0)
    
    for file_Number in pbar:
        
        # generates potentials and lables, sorts them into seperate arrays to be exported
        # also generates file name
        out, label, fileName = Sorter(image_size, num_pot_sho, num_pot_iw, num_pot_dig, num_pot_rnd, False, v, solve ) 
        
        full_fileName = fileNameStart + str(image_size) + fileName + str(file_Number + 100)
        save_path = os.path.join(path, full_fileName)
        
        # actualy creates a file & saves data to it 
        
        np.savez(save_path, images=out, labels = label) # writes to a .npz file
        
        pbar.set_description("Creating File " + full_fileName )
        #np.save((save_path + '.npy'), pot_dict) # write the potentials to a .npy file
        #print statements

        if v>=2:
            print('File [{}/{}] Generated'.format(file_Number, num_files))
            if v>=3:
                print("------------------")

    if v>=1:
        print("{:.3f} Minutes".format((time.time() - start_time)/60))

main()

