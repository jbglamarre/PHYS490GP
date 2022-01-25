# =============================================================================
# Imports
# =============================================================================

import sys , argparse , json , os , time
import numpy as np
import torch as tc
import torch.nn as nn
import torch.optim as op
import matplotlib.pyplot as plt
import tkinter as tk #Basic GUI
sys.path.append( "auxiliary" )
from tkinter import filedialog
from nn_gen import Net

 
# =============================================================================
# Functions
# =============================================================================


#-------------------------------AskForFile------------------------------------


def AskForPath(v): 
    '''
    prompts user to identify a file path containing data 
    
    input: v int - verbosity
    output: the selected directory
    '''
    
    root = tk.Tk() # make tk object
    root.withdraw()
    fpath= filedialog.askdirectory() 
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
        if filename.endswith(".npz"):
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
    
    images = tc.Tensor(data['images']).unsqueeze(1)
    labels = tc.Tensor(data['labels']).view(100,1)
    
    return images, labels


#---------------------------------print1--------------------------------------    


def print1(model,ifile, num_files, iepoch, num_epochs, num_epochs_display, verbosity , losses):
    '''
    This function generalizes print statements

    Arguments:
        - model: The network itself.
        - ifile: the current file number of the epoch run
        - num_files: The total number of files used
        - iepoch: The current epoch of the training run
        - num_epochs: The total expected number of epochs
        - num_epochs_display: what epochs do you want displayed
        - verbosity: The verbosity of the code
        - epoch_losses containes a list of loss'

    Returned values: None
    '''    
    predicted_count = model.predicted_count
    total_count     = model.total_count
    accuracy        = model.accuracy

    if ( verbosity == 2 ) and not ( iepoch + 1 ) % num_epochs_display:
        print( "Epoch: [%d/%d]\nLoss: %.7e\nAcccuracy: %.f %%\nCorrect: [%d/%d]" % (
            iepoch + 1 , num_epochs , losses[ iepoch ] ,
            accuracy , predicted_count , total_count ) )
        print('---------------------------------')
    
    elif ( verbosity >= 3 ) and not ( iepoch + 1 ) % num_epochs_display:
        print( "file: [%d/%d]\nLoss: %.7e\nAcccuracy: %.f %%\nCorrect: [%d/%d]"% ( 
            ((iepoch)*num_files) + ifile  , num_epochs*num_files , losses[ ((iepoch)*num_files) + ifile - 1 ] ,
            accuracy , predicted_count , total_count ) )
        print('---------------------------------')
     
        
#---------------------------------plotter-------------------------------------
    

def plotter(losses, num_itters, verbosity, x_label):
    '''
    This function plots the loss.

    Arguments:

        - epoch_losses: the losses from a single epoch's training/testing run
        - num_itter: The total expected number of epochs times the number of files per epoch
        - verbosity: The verbosity of the code
    '''

    # Plots average training and test losses over all epochs, then saves the figure.
    plt.plot( range( 1 , num_itters+1 ) , losses , c = "r" , label = "Loss" )
    plt.ylabel( "Loss")
    plt.xlabel( x_label )
    plt.legend( )
   # plt.savefig( "results/Epoch_Loss.pdf" )

    if verbosity >= 2:
        plt.show( )

    plt.close( )


#---------------------------------Exported-------------------------------------

# Exports the given data to the given path 
def Exporter(data,path):
    
    try:
        newTxtFile = open(path,"a+")
        
    except:
        print(" Could not open file: " + str(path) +'\n')
        
    for val in data:
        newTxtFile.write(str(val) + "\n")
        
    newTxtFile.close()
    
#---------------------------------Train----------------------------------------    


def Train( model, optimizer , criterion, dictionary, loss_save_path):
 
    '''
    This function trans the network.

    Arguments:

        - model: The network itself.
        - optimizer: The optimizer.
        - criterion: The loss calculator.
        - dictionary: dictionary obtained form the json file.

    Returned values:
    '''
    
    files_per_batch    = dictionary[ "files_per_batch"    ]
    num_epochs         = dictionary[ "num_epochs"         ]
    num_epochs_display = dictionary[ "num_epochs_display" ]
    verbosity          = dictionary[ "verbosity"          ]

    
    # get the path to the data from user
    file_Path = Get_File_Paths( )
    
     # Training 
    train_epoch_losses = [ ]
    train_file_losses = [ ]

    print("Start Training")
    # begin training loop
    for iepoch in range( 0 , num_epochs ):

        ifile = 1

        for path in file_Path:
            if verbosity>=3:
                print("Training on file: " + str(path))
            # load the images and lables for the specified files
            images, labels = Data_Loader( path, files_per_batch )
            
            # Training 
            model.trainer( optimizer , criterion , images, labels , train_file_losses )
            
            if verbosity>=3:
                # print staements
                print1(model, ifile * files_per_batch , len(file_Path)*files_per_batch , iepoch, num_epochs , num_epochs_display , verbosity , train_file_losses)
            
            ifile +=1
            
        train_epoch_losses.append( np.mean( train_file_losses[ (iepoch)*(ifile-1): (iepoch+1)*(ifile-1) ] ) )  
            
        if verbosity == 2:
            print1(model, ifile * files_per_batch , len(file_Path)*files_per_batch , iepoch, num_epochs , num_epochs_display , verbosity , train_epoch_losses)
            
    
    # Export the training epoch losses to a txt file
    Exporter( train_epoch_losses , loss_save_path )
    
    # plot the loss 
    if verbosity == 2:
        
        plotter( train_epoch_losses , num_epochs  , verbosity, "# of epochs" )
         
    elif verbosity >= 3:
        
        plotter( train_file_losses , num_epochs * len(file_Path) , verbosity, "# of files used" )
        
    
        
#---------------------------------Test----------------------------------------       


def Test( model, optimizer , criterion, dictionary, performance_save_path):
    '''
    This function tests the network.

    Arguments:

        - model: The network itself.
        - optimizer: The optimizer.
        - criterion: The loss calculator.
        - dictionary: dictionary obtained form the json file.

    Returned values:
    '''
    
    files_per_batch    = dictionary[ "files_per_batch"    ]
    num_epochs_display = dictionary[ "num_epochs_display" ]
    verbosity          = dictionary[ "verbosity"          ]
    # get the path to the data from user
    test_File_Path = Get_File_Paths( )
    
    # Training 
    test_losses = [ ]
    
    performance = [ ]
    
    ifile = 1
    
    for path in test_File_Path :
        if verbosity>=3:
            print("Testing on file: " + str(path))
        
        # load the images and lables for the specified files
        images, labels = Data_Loader( path, files_per_batch )
        
        # Testing
        out = model.tester( optimizer , criterion , images, labels , test_losses )
        
        if verbosity>=3:
            # print statements
            print1(model, ifile * files_per_batch , len(test_File_Path) * files_per_batch , 0 , 1 , num_epochs_display , verbosity , test_losses )
            
        performance+=list(zip(np.array(out)[:,0],np.array(labels)[:,0]))
        
        ifile +=1
        
    Exporter( performance , performance_save_path )  
    
    # plot the losses    
    if verbosity >= 2:
        plotter( test_losses, len(test_File_Path) , verbosity, "# of files used" )
        

# =============================================================================
# main function
# =============================================================================

def main( ):
    '''
        This executes the main classification process.
    '''
    
    # Adds and reads in command line arguments
    parser = argparse.ArgumentParser( description = "main.py Arguments." )
    
    args           = parser.parse_args( )
    
    args.pf = "param/param_filename.json"
    param_filename = args.pf

    # Reads in .json file parameters.
    with open( param_filename ) as dictionary_file:
        dictionary            = json.load( dictionary_file )
        learning_rate         = dictionary[ "learning_rate"        ]
        model_load_path       = dictionary[ "model_load_path" ]
        model_save_path       = dictionary[ "model_save_path" ]
        loss_save_path        = dictionary[ "loss_save_path" ]
        performance_save_path = dictionary[ "performance_save_path" ]
        train                 = dictionary[ "train" ]
        test                  = dictionary[ "test"]
	 
        
    # Initializes model.
    model     = Net( 6 ).to( tc.device( "cpu" ) )

    optimizer = op.Adam( model.parameters( ) , lr = learning_rate )
    
    criterion = nn.MSELoss( )

    # Resets previously trained parameters.
    model.reset( )

    if train:
        if os.path.exists(model_load_path):
            model.load_state_dict(tc.load(model_load_path))
            print("Loading model: " + str(model_load_path))
            
        elif dictionary[ "verbosity" ]>=1:
            print("An existing model of name: " + str(model_load_path) +' does not exist. The Model will be saved to a new mdoel instance with the same name')
            
        
        Train( model, optimizer , criterion, dictionary , loss_save_path)
        
        tc.save(model.state_dict(), model_save_path)
        
    if test:
        try:
            #Load the model weights from testing phase 
            model.load_state_dict(tc.load(model_load_path))
            print("Loading model: " + str(model_load_path))
            
        except:
            print("An existing model of name: " + str(model_load_path) +' does not exist')
          
        Test( model, optimizer , criterion, dictionary , performance_save_path)
        
# -----------------------------------------------------------------------------
# main call
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Stores start time.
    start_time = time.time( )

    # Runs main function.
    main( )

    # Stores end time.
    end_time   = time.time( )

    # Computes total runtime.
    total_time = time.strftime( "%H hours, %M minutes, and %S seconds" , time.gmtime( end_time - start_time ) )

    # Prints runtime.
    print( "\n[Finished in %s.]\n" % total_time )
 
