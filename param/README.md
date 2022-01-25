This will be the file containnig the pertinent information for a run of the main function. Arguments include:

"files_per_batch"    : Int - Should be left as 1 - How many files to use per batch

"learning_rate"      : float - Learning rate for Adam optimizer

"num_epochs" 		     : Int - Number of epochs to train on

"num_epochs_display" : Int - How jmany epochs do you want displayed updates on 

"verbosity"          : Int - 1,2 or 3

"model_load_path"    : String - Path - Path to and name of file where the trained weights will be loaded from

"model_save_path"    : String - Path - Path to and name of file where the trained weights will be storred

"loss_save_path"     : String - Path (includes .txt) - Path to and name of the file to save the loss 

"performance_save_path" :   String - Path (includes .txt) - Path to and name of the file to save the testing performance of the network on testing data

"train"              : Boolean - To turn off or on training

"test"               : Boolean - To turn off or on testing
