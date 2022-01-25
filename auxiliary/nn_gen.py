# =============================================================================
# Imports
# =============================================================================

import torch as tc
import torch.nn as nn
import torch.nn.functional as fu

#  ---------------------------------------------------------------------------------------------------------------------------------------------
# Class: Net
# ---------------------------------------------------------------------------------------------------------------------------------------------

class Net(nn.Module):
    '''
    Deep neural network class.
    Architecture:
        7 reducing convolutional layers with kernel size 3x3 pixels, 64 filters
        and 2x2 stride.
        2 non-reducing convolutional layers  with kernel size 4x4, 16 filters,
        and unit stride 1 in between each reducing layer for a total of 12.
        2 fully conected layers at the end of width 1024 and 1 respectively.
        All layers have RelU activation
    '''

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    # ---------------------------------------------------------------------------------------------------------------------------------------------

    def __init__( self , n_convo ):
        super( Net , self ).__init__( )
        
        #Initiate deep neural network
        self.layers = nn.ModuleList( )
        self.layers.append(nn.Conv2d( 1 , 64 , 3 , stride = 2 ) )
        for i in range( n_convo - 1 ):
            self.layers.append(nn.Conv2d( 64 , 16 , 4 , stride = 1 ) )
            self.layers.append(nn.Conv2d( 16 , 16 , 4 , stride = 1 ) )
            self.layers.append(nn.Conv2d( 16 , 64 , 3 , stride = 2 ) )
        self.layers.append(nn.Linear( 64*2*2 , 1024 ) )
        self.layers.append(nn.Linear( 1024 ,1 ) )
        
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Function: forward
    # ---------------------------------------------------------------------------------------------------------------------------------------------

    def forward( self , x ):
        
        layer_count = 0
        for layer in self.layers[ : - 2 ]:

            if layer_count % 3 == 0:
                x = fu.pad( x , ( 0 , 1 , 0 , 1 ) , mode = "replicate" )
            else:
                x = fu.pad( x , ( 1 , 2 , 1 , 2 ) , mode = "replicate" )

            layer_count += 1

            x = fu.relu( layer( x ) )
            
        x = x.view(-1, 64*2*2)

        x = fu.relu( self.layers[ - 2 ]( x ) )
        
        x = self.layers[ - 1 ]( x )
        
        return x
        
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Function: reset
    # ---------------------------------------------------------------------------------------------------------------------------------------------

    def reset(self):
        for layer in self.layers:
            layer.reset_parameters()

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Function: trainer
    # ---------------------------------------------------------------------------------------------------------------------------------------------

    def trainer( self ,  optimizer , criterion , images, labels , file_losses ):
        '''
            This function trains the network.

            Arguments:

                - model: The network itself.
                - optimizer: The optimizer.
                - criterion: The loss calculator.
                - loader: The dataloader object (dtype = dataset).
                - epoch_losses: The list of average losses for each epoch (dtype = list).

            Returned values:

                - predicted_count: Number of correctly labelled images (dtype = int).
                - total_count: Number of images (dtype = int).
                - accuracy: Accuracy of the network (dtype = float).
        '''

        # Computes and appends batch's loss.
        outputs = self.forward( images )
        loss    = criterion(outputs , labels )
        #print(outputs)
        
        # Sums batch's number of correctly labelled and total images.
        predicted_count = tc.count_nonzero( tc.isclose(outputs, labels, atol=1e-3, equal_nan=False) ).item()

        # Sets learning settings for next batch.
        optimizer.zero_grad( )
        loss.backward( )
        optimizer.step( )
        
        # Computes epoch's average loss and accuracy.
        file_losses.append( loss.item( ) )


        # Allows values to be called.
        self.predicted_count = predicted_count
        self.total_count     = labels.size( 0 )
        self.accuracy        = 100 * predicted_count / self.total_count


    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Function: tester
    # ---------------------------------------------------------------------------------------------------------------------------------------------

    def tester( self , optimizer , criterion , images, labels  , losses ):
        '''
            This function tests the network without the calculation of gradients.

            Arguments:

                - model: The network itself.
                - optimizer: The optimizer.
                - criterion: The loss calculator.
                - loader: The dataloader object (dtype = dataset).
                - epoch_losses: The list of average losses for each epoch (dtype = list).

            Returned values:

                - predicted_count: Number of correctly labelled images (dtype = int).
                - total_count: Number of images (dtype = int).
                - accuracy: Accuracy of the network (dtype = float).
        '''

        # Initializes variables.

        with tc.no_grad( ): # Turns of gradient calculations.

            # Computes and appends batch's loss.
            outputs = self.forward( images )
            loss    = criterion( self.forward( images ) , labels )
            
            # Sums batch's number of correctly labelled and total images.
            predicted_count = tc.count_nonzero( tc.isclose(outputs, labels, atol=1e-1, equal_nan=False) ).item()
            
            # Computes epoch's average loss and accuracy.
            losses.append( loss.item( ) )

            # Allows values to be called.
            self.predicted_count = predicted_count
            self.total_count     = labels.size( 0 )
            self.accuracy        = 100 * predicted_count / self.total_count
            
        return outputs