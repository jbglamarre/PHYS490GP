import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pylab import cm

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Class: potentials_gen
# -----------------------------------------------------------------------------------------------------------------------------------------------------

class plots_gen( ):
	'''
		This class generates the loss and true vs predicted potential plots.
	'''

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: __init__
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def __init__( self ):
		'''
			This function initializes the class.

			Arguments:

				- self: Class instance.
		'''

		self.potentials = [ "SHO" , "IW" , "DIG" , "RND" ]

		return( None )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: loss_plotter
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def loss_plotter( self , sho_loss_filename , iw_loss_filename , dig_loss_filename , rnd_loss_filename ):
		'''
			This function generates the loss plot.

			Arguments:

				- self: Class instance.
				- sho_loss_filename: The filename for the simple harmonic oscillator model (dtype = str).
				- iw_loss_filename: The filename for the infinite well model (dtype = str).
				- dig_loss_filename: The filename for the double-well inverted Gaussian model (dtype = str).
				- rnd_loss_filename: The filename for the random model (dtype = str).
		'''

		# Sets up colors.
		cmap = cm.get_cmap( "magma" , 8 )
		colors = [ cmap( 0 ) , cmap( 2 ) , cmap( 3 ) , cmap( 4 ) ]

		# Loss filenames list.
		loss_filenames = [ sho_loss_filename , iw_loss_filename , dig_loss_filename , rnd_loss_filename ]

		# Loops through potential types.
		for ipot in range( 0 , len( self.potentials ) ):

			# Loads in loss file.
			loss = np.loadtxt( "../results/%s" % loss_filenames[ ipot ] )

			# Initializes epoch range.
			if ipot == 2:
				num_epochs = np.linspace( 1 , int( len( loss ) / 2 ) , len( loss ) )
			else:
				num_epochs = range( 1 , len( loss ) + 1 )

			# PLots log log plot of loss vs epochs.
			plt.loglog( num_epochs , loss , color = colors[ ipot ] , label = self.potentials[ ipot ] , lw = 2 )

		# Sets plot parameters.
		plt.legend( )
		plt.xlabel( "Epoch" , fontsize = 14 )
		plt.ylabel( "Training Loss" , fontsize = 14 )
		plt.show( )
		plt.close( )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: performance_plotter
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def performance_plotter( self , sho_performance_filename , iw_performance_filename , dig_performance_filename , rnd_performance_filename ):
		'''
			This function generates the loss plot.

			Arguments:

				- self: Class instance.
				- sho_loss_filename: The filename for the simple harmonic oscillator model (dtype = str).
				- iw_loss_filename: The filename for the infinite well model (dtype = str).
				- dig_loss_filename: The filename for the double-well inverted Gaussian model (dtype = str).
				- rnd_loss_filename: The filename for the random model (dtype = str).
		'''

		# Performance filenames list.
		performance_filenames = [ sho_performance_filename , iw_performance_filename , dig_performance_filename , rnd_performance_filename ]

		# Main plot axis limits.
		ax1_lim = [ 100 , 400 ]
		ax2_lim = [ - 10 , 10 ]

		def gaussian( x , a , mu , sigma ):
			'''
				This function generates the loss plot.

				Arguments:

					- x: The x-values (dtype = nd.array).
					- a: The Gaussian's amplitude (dtype = float).
					- mu: The Gaussian's mean (dtype = float).
					- sigma: The Gaussian standard deviation (dtype = float).
			'''

			return( a * np.exp( - ( ( x - mu ) / ( np.sqrt( 2 ) * sigma ) ) ** 2 ) )

		# Loops through potential types.
		for ipot in range( 0 , len( self.potentials ) ):

			# Sets scale.
			if ipot == 2 or ipot == 3:
				scale = 100
			else:
				scale = 1000

			# Loads in performance file.
			performance = np.loadtxt( "../results/%s" % performance_filenames[ ipot ] , dtype = tuple )

			# Formats data.
			performance_0 = [ ]
			performance_1 = [ ]			
			for irow in range( 0 , len( performance[ : , 0 ] ) ):
				performance[ irow , 0 ] = eval( performance[ irow , 0 ].replace( "(" , "" ).replace( "," , "" ) )
				performance[ irow , 1 ] = eval( performance[ irow , 1 ].replace( ")" , "" ) )

				if ( performance[ irow , 0 ] >= ( ax1_lim[ 0 ] / scale ) ) * ( performance[ irow , 0 ] <= ( ax1_lim[ 1 ] / scale ) ) * ( performance[ irow , 1 ] >= ( ax1_lim[ 0 ] / scale ) ) * ( performance[ irow , 1 ] <= ( ax1_lim[ 1 ] / scale ) ) == 1:
					performance_0.append( performance[ irow , 0 ] )
					performance_1.append( performance[ irow , 1 ] )

			# Scales performance values.
			performance = np.array( [ performance_1 , performance_0 ] ).T * scale

			# Initializes figure.
			fig , ax1 = plt.subplots( )

			# Plots main plot histogram.
			hist2d = ax1.hist2d( *performance.T , bins = ax1_lim[ 1 ] - ax1_lim[ 0 ] , cmap = "magma" , cmin = 1 )
			
			# Determines counts.
			counts , _ , _ = np.histogram2d( *performance.T , bins = ax1_lim[ 1 ] - ax1_lim[ 0 ] )

			# Sets colorbar.
			fig.colorbar( hist2d[ 3 ] , ticks = [ 1 , int( np.max( counts ) / 2 ) , np.max( counts ) ] )

			# Sets plot paremeters.
			ax1.plot( np.linspace( ax1_lim[ 0 ] , ax1_lim[ 1 ] , 1000 ) , 
					  np.linspace( ax1_lim[ 0 ] , ax1_lim[ 1 ] , 1000 ) , 
					  color = ( 0 , 1 , 0 , 1 ) , lw = 0.7 )
			ax1.set_xlim( *ax1_lim )
			ax1.set_ylim( *ax1_lim )
			ax1.set_xticks( np.arange( ax1_lim[ 0 ] , ax1_lim[ 1 ] + 1 , 100 ) )
			ax1.set_yticks( np.arange( ax1_lim[ 0 ] , ax1_lim[ 1 ] + 1 , 100 ) )

			if ipot == 2 or ipot == 3:
				ax1.set_xlabel( r"True Energy [mHa $\times$ 10]" , fontsize = 14 )
				ax1.set_ylabel( r"Predicted Energy [mHa $\times$ 10]" , fontsize = 14 )
			else:
				ax1.set_xlabel( "True Energy [mHa]" , fontsize = 14 )
				ax1.set_ylabel( "Predicted Energy [mHa]" , fontsize = 14 )

			ax1.grid( )

			# Initializes inset plot.
			ax2 = inset_axes( ax1 , width = "30%" , height = "30%" , loc = 4 )

			# Computes error and MAE.
			error = performance[ : , 0 ] - performance[ : , 1 ]
			MAE = np.median( np.abs( error - np.median( error ) ) )

			if ipot == 2 or ipot == 3:
				print( 10 * MAE )
			else:
				print( MAE )

			# Sets step and yscale.
			step = 0.1
			yscale = 100

			# PLots inset histogram.
			ax2.hist( error , bins = np.arange( np.min( error ) , np.max( error ) + step , step ) , color = "k" )

			# Computes histogram counts and positions.
			counts , positions = np.histogram( error , bins = np.arange( np.min( error ) , np.max( error ) + step , step ) )

			# Sets ax2_x and ax2_y.
			ax2_x = np.linspace( ax2_lim[ 0 ] , ax1_lim[ 1 ] , int( 1000 / step ) )

			if ipot == 2:
				ax2_y = yscale * np.arange( 0 , ( np.floor( np.max( counts ) / ( yscale * 1 ) ) * 1 ) + 1 + 0.1 , 1 )	
			if ipot == 3:
				ax2_y = yscale * np.arange( 0 , 11 , 5 )
			else:
				ax2_y = yscale * np.arange( 0 , int( np.round( np.max( counts ) / yscale , - 1 ) + 6 ) , 5 )

			# Fits gaussian to inset histogram.
			popt , pcov = curve_fit( gaussian , np.arange( np.min( error ) , np.max( error ) , step ) , 
									 counts , p0 = [ np.max( counts ) , np.mean( error ) , np.std( error ) ] )

			# Plots Gaussian fit.
			ax2.plot( ax2_x + ( step / 2 ) , gaussian( ax2_x , *popt ) , color = ( 0 , 1 , 0 , 1 ) , lw = 1.2 )

			# Sets plot parameters.
			ax2.fill_between( ax2_x + ( step / 2 ) , np.max( ax2_y ) , color = "k" , alpha = 0.1 )

			ax2.set_xlim( ax2_lim[ 0 ] , ax2_lim[ 1 ] )
			ax2.set_xticks( [ ax2_lim[ 0 ] , popt[ 1 ] + ( step / 2 ) , ax2_lim[ 1 ] ] )
			ax2.set_xticklabels( [ ax2_lim[ 0 ] , "%.1f" % ( popt[ 1 ] + ( step / 2 ) ) , ax2_lim[ 1 ] ] )

			ax2.set_ylim( np.min( ax2_y ) , np.max( ax2_y ) )
			ax2.set_yticks( ax2_y )

			ax2.set_yticklabels( np.array( ax2_y / yscale , dtype = int ) )

			if ipot == 2 or ipot == 3:
				ax2.set_xlabel( r"Error [mHa $\times$ 10]" , fontsize = 12 )
			else:
				ax2.set_xlabel( "Error [mHa]" , fontsize = 14 )

			ax2.set_ylabel( r"Count $\left[10^2\right]$" , fontsize = 14 )
			ax2.xaxis.set_ticks_position( "top" ) 
			ax2.xaxis.set_label_position( "top" )

			fig.suptitle( "%s" % self.potentials[ ipot ] , fontsize = 16 )

			plt.tight_layout( )
			plt.show( )
			plt.close( )

# Runs plot generators.
plots_gen( ).loss_plotter( "Training_Loss_1000_100_SHO.txt" , "Training_Loss_1000_100_IW.txt" , "Training_Loss_500_125_DIG.txt" , "Training_Loss_1000_100_RND.txt" )
plots_gen( ).performance_plotter( "Performance_250_100_SHO.txt" , "Performance_250_100_IW.txt" , "Performance_250_100_DIG.txt" , "Performance_250_100_RND.txt" )
