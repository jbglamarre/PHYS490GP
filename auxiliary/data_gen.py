# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------
 
import numpy as np
import time as tm
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from cv2 import resize # pip install opencv-python
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon # pip install shapely
from numba import jit # pip install numba

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Class: potentials
# -----------------------------------------------------------------------------------------------------------------------------------------------------

class potentials( ):
	'''
		This class generates the potential functions.
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

		return( None )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: sho_potential
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def sho_potential( self , x , y , c_x , c_y , k_x , k_y ):
		'''
			This function generates the simple harmonic oscillator potential.

			Arguments:

				- self: Class instance.
				- x: x data in a.u. (dtype = numpy.ndarray).
				- y: y data in a.u. (dtype = numpy.ndarray).
				- c_x: Center x coordinate (dtype = float).
				- c_y: Center y coordinate (dtype = float).
				- k_x: x-axis spring constant (dtype = float).
				- k_y: y-axis spring constant (dtype = float).

			Returned values:

				- V: The potential given the arguments (dtype = numpy.ndarray).
		'''

		# Simple harmonic oscillator potential function.
		V = 0.5 * ( ( k_x * ( ( x - c_x ) ** 2 ) ) + ( k_y * ( ( y - c_y ) ** 2 ) ) )

		# Truncates potentials greater than 20 to 20.
		V = ( 20 * ( V > 20 ) ) + V * ( V <= 20 )

		return( V )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: iw_potential
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def iw_potential( self , x , y , c_x , c_y , L_x , L_y ):
		'''
			This function generates the infinite well potential.

			Arguments:

				- self: Class instance.
				- x: x data in a.u. (dtype = numpy.ndarray).
				- y: y data in a.u. (dtype = numpy.ndarray).
				- c_x: Well center x coordinate (dtype = float).
				- c_y: Well center y coordinate (dtype = float).
				- L_x: Well x-axis width (dtype = float).
				- L_y: Well y-axis width (dtype = float).

			Returned values:

				- V: The potential given the arguments (dtype = numpy.ndarray).
		'''

		# Sets well bounds.
		lb_x = 0.5 * ( ( 2 * c_x ) - L_x )
		ub_x = 0.5 * ( ( 2 * c_x ) + L_x )
		lb_y = 0.5 * ( ( 2 * c_y ) - L_y )
		ub_y = 0.5 * ( ( 2 * c_y ) + L_y )

		# Infinite potential well potential function.
		V = 20 * ( ( x <= lb_x ) + ( x > ub_x ) + ( y <= lb_y ) + ( y > ub_y ) )

		return( V )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: dig_potential
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def dig_potential( self , x , y , A_1 , A_2 , c_x_1 , c_y_1 , c_x_2 , c_y_2 , k_x_1 , k_y_1 , k_x_2 , k_y_2 ):
		'''
			This function generates the double-well inverted Gaussian potential.

			Arguments:

				- self: Class instance.
				- x: x data in a.u. (dtype = numpy.ndarray).
				- y: y data in a.u. (dtype = numpy.ndarray).
				- A_1: Well 1 depth (dtype = float).
				- A_2: Well 2 depth (dtype = float).
				- c_x_1: Well 1 center x coordinate (dtype = float).
				- c_y_1: Well 1 center y coordinate (dtype = float).
				- c_x_dea2: Well 2 center x coordinate (dtype = float).
				- c_y_2: Well 2 center y coordinate (dtype = float).
				- k_x_1: Well 1 x-axis width (dtype = float).
				- k_y_1: Well 1 y-axis width (dtype = float).
				- k_x_2: Well 2 x-axis width (dtype = float).
				- k_y_2: Well 2 y-axis width (dtype = float).

			Returned values:

				- V: The potential given the arguments (dtype = numpy.ndarray).
		'''

		# Well potentials.
		V_1 = - A_1 * np.exp( ( - ( ( ( x - c_x_1 ) / k_x_1 ) ** 2 ) ) + ( - ( ( ( y - c_y_1 ) / k_y_1 ) ** 2 ) ) )
		V_2 = - A_2 * np.exp( ( - ( ( ( x - c_x_2 ) / k_x_2 ) ** 2 ) ) + ( - ( ( ( y - c_y_2 ) / k_y_2 ) ** 2 ) ) )
		
		# Double-well inverted Gaussian potential function
		V = V_1 + V_2

		# Translation to zero minimum and scaling to 20 maximum.
		V = V + max( A_1 , A_2 )
		V = ( 20 / np.max( V ) ) * V

		return( V )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: rnd_potential
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def rnd_potential( self , x , y , sigma_1 , k , R , sigma_2 , d , image_size ):
		'''
			This function generates the random potential.

			Arguments:

				- self: Class instance.
				- x: x data in a.u. (dtype = numpy.ndarray).
				- y: y data in a.u. (dtype = numpy.ndarray).
				- sigma_1: First Gaussian blur standard deviation floats (dtype = numpy.ndarray).
				- k: Random resolution integers (dtype = numpy.ndarray).
				- R: Random resolution integers (dtype = numpy.ndarray).
				- sigma_2: Second Gaussian blur standard deviation floats (dtype = numpy.ndarray).
				- d: Random exponentiation floats from list of values (dtype = numpy.ndarray).
				- image_size: Image dimensions (dtype = int).

			Returned values:

				- V: The potential given the arguments (dtype = numpy.ndarray).
		'''

		# Generate and resize first binary grid.
		grid_1 = np.array( np.random.default_rng( ).integers( 0 , 2 , [ 16 , 16 ] ) , dtype = "uint8" )
		grid_1 = np.array( resize( grid_1 , dsize = ( image_size , image_size ) ) , dtype = "float64" )

		# Generate and resize second binary grid.
		grid_2 = np.array( np.random.default_rng( ).integers( 0 , 2 , [ 16 , 16 ] ) , dtype = "uint8" )
		grid_2 = np.array( resize( grid_2 , dsize = ( int( image_size / 2 ) , int( image_size / 2 ) ) ) , dtype = "float64" )
		grid_2 = np.pad( grid_2 , int( image_size / 4 ) , mode = "constant" , constant_values = 0 )

		# Subtract the two grids, then blur and exponentiate results.
		grid = abs( grid_1 - grid_2 )
		grid = gaussian_filter( grid , sigma = sigma_1 )
		grid = grid ** d

		# Generate blob grid.
		blob = np.ones( [ image_size , image_size ] )

		# Randomly generate convex hull and return its path.
		blob_size 	= 200
		blob_center = ( ( image_size - blob_size ) / 2 )
		blob_points = np.random.default_rng( ).integers( blob_center - 1 , blob_size + blob_center , [ k ** 2 , 2 ] )
		hull 	  	= ConvexHull( blob_points )
		hull_path 	= blob_points[ hull.vertices , : ]
		hull_path 	= np.vstack( [ hull_path , hull_path[ 0 , : ] ] )

		# Interpolate convex hull path.
		knots , u 				  = splprep( hull_path.T , k = 3 , per = 1 )
		hull_path_x , hull_path_y = splev( np.linspace( min( u ) , max( u ) , 1000 ) , knots , der = 0 )
		hull_path 				  = np.vstack( [ hull_path_x , hull_path_y ] ).T

		# Locate the convex hull path's centroid (center of mass).
		hull_center = Polygon( hull_path ).centroid.coords
		hull_center = hull_center[ 0 ]

		# Compute differences between centroid and grid center.
		hull_center_x = ( image_size / 2 ) - hull_center[ 0 ] - 1
		hull_center_y = ( image_size / 2 ) - hull_center[ 1 ] - 1

		# Recenter convec hull path to grid center.
		hull_path_x = hull_path[ : , 0 ] + hull_center_x
		hull_path_y = hull_path[ : , 1 ] + hull_center_y
		hull_path 	= np.vstack( [ hull_path_x , hull_path_y ] ).T
		hull_path 	= Path( hull_path )

		# Set blob grid points within convex hull path to 1 and those outside to 0, creating a binary mask.
		blob_indices = np.vstack( [ np.where( blob >= 0 )[ 0 ] , np.where( blob >= 0 )[ 1 ] ] ).T
		blob 		 = 1 * ( hull_path.contains_points( blob_indices ).reshape( image_size , image_size ) )
		blob 		 = blob.T

		# Resize blob to R x R resolution.
		blob = np.array( blob , dtype = "uint8" )
		blob = np.array( resize( blob , dsize = ( R , R ) ) , dtype = "float64" )

		# Pad blob edges with 0s for image_size x image_size resolution.
		if R % 2 == 0:
			blob = np.pad( blob , int( ( image_size - R ) / 2 ) , mode = "constant" , constant_values = 0 )
		elif R % 2 == 1:
			blob = np.pad( blob , int( ( image_size - R + 1 ) / 2 ) , mode = "constant" , constant_values = 0 )
			blob = blob[ : - 1 , : - 1 ]

		# Blur blob such that it smoothly goes from 1 in the interior to 0 at the edges.
		blob = gaussian_filter( blob , sigma = sigma_2 )

		# Multiplication of original grid and the blob mask, as well as inversion and scaling to 20 maximum.
		V = grid * blob
		V = ( np.max( V ) * np.ones( [ image_size , image_size ] ) ) - V
		V = ( 20 / np.max( V ) ) * V

		return( V )

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Class: potentials_gen
# -----------------------------------------------------------------------------------------------------------------------------------------------------

class potentials_gen( ):
	'''
		This class generates batches of potentials.

		Note that each potential_gen function generates num_pot images and outputs an array of dimension: 
		( num_pot , image_size , image_size ).
	'''

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: __init__
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def __init__( self , image_size , num_pot ):
		'''
			This function initializes the class.

			Arguments:

				- self: Class instance.
				- image_size: Image dimensions (dtype = int).
				- num_pot: Number of potentials to generate (dtype = int).

			Returned values:

				- image_size: Image dimensions (dtype = int).
				- num_pot: Number of potentials to generate (dtype = int).
				- x: x data in a.u. (dtype = numpy.ndarray).
				- y: y data in a.u. (dtype = numpy.ndarray).
		'''

		# Sets attributes.
		self.image_size = image_size
		self.num_pot 	= num_pot
		
		self.x , self.y = np.meshgrid( np.linspace( - 20 , 20 , image_size ) , np.linspace( - 20 , 20 , image_size ) )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: potential_plot_gen
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def potential_plot_gen( self , plot , num_pot , x , y , V_batch ):
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

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: sho_potential_gen
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def sho_potential_gen( self , plot = False ):
		'''
			This function generates simple harmonic oscillator potentials.

			Arguments:

				- self: Class instance.
				- plot: Show 3D and 2D plots of this potential of this type (dtype = bool).

			Returned values:

				- c_x: Center x coordinate (dtype = float).
				- c_y: Center y coordinate (dtype = float).
				- k_x: x-axis spring constant (dtype = float).
				- k_y: y-axis spring constant (dtype = float).
				- V_batch: num_pot simple harmonic oscillator potentials (dtype = numpy.ndarray).
		'''

		# ---------------------------------------------------------------------------------------------------------------------------------------------
		# Function: sho_loop_gen
		# ---------------------------------------------------------------------------------------------------------------------------------------------

		@jit( nopython = True ) # Translates function to machine language in order to compile quickly. 
		def sho_loop_gen( num_pot , image_size , x , y , c_x , c_y , k_x , k_y ):
			'''
				This executes the potential batch generation loop for simple harmonic oscillator potentials.

				Arguments:

					- image_size: Image dimensions (dtype = int).
					- num_pot: Number of potentials to generate (dtype = int).
					- x: x data in a.u. (dtype = numpy.ndarray).
					- y: y data in a.u. (dtype = numpy.ndarray).
					- c_x: Center x coordinate (dtype = float).
					- c_y: Center y coordinate (dtype = float).
					- k_x: x-axis spring constant (dtype = float).
					- k_y: y-axis spring constant (dtype = float).

				Returned values:

					- V_batch: num_pot simple harmonic oscillator potentials (dtype = numpy.ndarray).
			'''

			# -----------------------------------------------------------------------------------------------------------------------------------------
			# Function: sho_potential
			# -----------------------------------------------------------------------------------------------------------------------------------------

			def sho_potential( x , y , c_x , c_y , k_x , k_y ):
				'''
					This function generates the simple harmonic oscillator potential.

					Arguments:

						- x: x data in a.u. (dtype = numpy.ndarray).
						- y: y data in a.u. (dtype = numpy.ndarray).
						- c_x: Center x coordinate (dtype = float).
						- c_y: Center y coordinate (dtype = float).
						- k_x: x-axis spring constant (dtype = float).
						- k_y: y-axis spring constant (dtype = float).

					Returned values:

						- V: The potential given the arguments (dtype = numpy.ndarray).
				'''

				# Simple harmonic oscillator potential function.
				V = 0.5 * ( ( k_x * ( ( x - c_x ) ** 2 ) ) + ( k_y * ( ( y - c_y ) ** 2 ) ) )

				# Truncates potentials greater than 20 to 20.
				V = ( 20 * ( V > 20 ) ) + V * ( V <= 20 )

				return( V )

			# Generates potential list.
			V_batch = [ ]

			for ipot in range( 0 , num_pot ):
				V_batch.append( sho_potential( x , y , c_x[ ipot ] , c_y[ ipot ] , k_x[ ipot ] , k_y[ ipot ] ) )

			return( V_batch )

		# Stores attributes.
		num_pot    = self.num_pot
		image_size = self.image_size
		x 		   = self.x
		y 		   = self.y

		# Generates c_x, c_y, k_x, and k_y arrays.
		c_x = np.random.default_rng( ).uniform( - 8 , 8 , num_pot )
		c_y = np.random.default_rng( ).uniform( - 8 , 8 , num_pot )
		k_x = np.random.default_rng( ).uniform( 0 , 0.16 , num_pot )
		k_y = np.random.default_rng( ).uniform( 0 , 0.16 , num_pot )

		# Sets attributes.		
		self.c_x = c_x
		self.c_y = c_y
		self.k_x = k_x
		self.k_y = k_y

		# Turns potential list to stacked potential array.

		V_batch = np.array( sho_loop_gen( num_pot , image_size , x , y , c_x , c_y , k_x , k_y ) ).reshape( num_pot , image_size , image_size )

		# Sets attributes.
		self.V_batch = V_batch

		# Generates figures for each potential with 3D and 2D subplots.
		self.potential_plot_gen( plot , num_pot , x , y , V_batch )

		# Returns class instance.
		return( self )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: iw_potential_gen
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def iw_potential_gen( self , plot = False ):
		'''
			This function generates infinite well potentials.

			Arguments:

				- self: Class instance.
				- plot: Show 3D and 2D plots of each potential of this type (dtype = bool).

			Returned values:

				- c_x: Well center x coordinate (dtype = float).
				- c_y: Well center y coordinate (dtype = float).
				- L_x: Well x-axis width (dtype = float).
				- L_y: Well y-axis width (dtype = float).
				- V_batch: num_pot infinite well potentials (dtype = numpy.ndarray).
		'''

		# ---------------------------------------------------------------------------------------------------------------------------------------------
		# Function: iw_loop_gen
		# ---------------------------------------------------------------------------------------------------------------------------------------------

		@jit( nopython = True ) # Translates function to machine language in order to compile quickly. 
		def iw_loop_gen( num_pot , image_size , x , y , c_x , c_y , L_x , L_y ):
			'''
				This executes the potential batch generation loop for infinite well potentials.

				Arguments:

					- image_size: Image dimensions (dtype = int).
					- num_pot: Number of potentials to generate (dtype = int).
					- x: x data in a.u. (dtype = numpy.ndarray).
					- y: y data in a.u. (dtype = numpy.ndarray).
					- c_x: Well center x coordinate (dtype = float).
					- c_y: Well center y coordinate (dtype = float).
					- L_x: Well x-axis width (dtype = float).
					- L_y: Well y-axis width (dtype = float).

				Returned values:

					- V_batch: num_pot infinite well potentials (dtype = numpy.ndarray).
			'''

			# -----------------------------------------------------------------------------------------------------------------------------------------
			# Function: iw_potential
			# -----------------------------------------------------------------------------------------------------------------------------------------

			def iw_potential( x , y , c_x , c_y , L_x , L_y ):
				'''
					This function generates the infinite well potential.

					Arguments:

						- x: x data in a.u. (dtype = numpy.ndarray).
						- y: y data in a.u. (dtype = numpy.ndarray).
						- c_x: Well center x coordinate (dtype = float).
						- c_y: Well center y coordinate (dtype = float).
						- L_x: Well x-axis width (dtype = float).
						- L_y: Well y-axis width (dtype = float).

					Returned values:

						- V: The potential given the arguments (dtype = numpy.ndarray).
				'''

				# Sets well bounds.
				lb_x = 0.5 * ( ( 2 * c_x ) - L_x )
				ub_x = 0.5 * ( ( 2 * c_x ) + L_x )
				lb_y = 0.5 * ( ( 2 * c_y ) - L_y )
				ub_y = 0.5 * ( ( 2 * c_y ) + L_y )

				# Infinite potential well potential function.
				V = ( x <= lb_x ) + ( x > ub_x ) + ( y <= lb_y ) + ( y > ub_y )
				V = 20 * V

				return( V )

			# Generates potential list.
			V_batch = [ ]

			for ipot in range( 0 , num_pot ):
				V_batch.append( iw_potential( x , y , c_x[ ipot ] , c_y[ ipot ] , L_x[ ipot ] , L_y[ ipot ] ) )

			return( V_batch )

		# Stores attributes.
		num_pot    = self.num_pot
		image_size = self.image_size
		x 		   = self.x
		y 		   = self.y

		# Generates c_x, c_y, E, and L_x arrays.
		c_x = np.random.default_rng( ).uniform( - 8 , 8 , num_pot )
		c_y = np.random.default_rng( ).uniform( - 8 , 8 , num_pot )
		E 	= np.random.default_rng( ).uniform( 0 , 0.4 , num_pot )
		L_x = np.random.default_rng( ).uniform( 4 , 15 , num_pot )

		# Solves for L_y array given E and L_x arrays.
		E_term 	 = np.zeros( num_pot ) 
		L_x_term = np.zeros( num_pot )

		for ipot in range( 0 , num_pot ):
			while E_term[ ipot ] <= L_x_term[ ipot ]:
				E[ ipot ]   = np.random.default_rng( ).uniform( 0 , 0.4 , 1 )
				L_x[ ipot ] = np.random.default_rng( ).uniform( 4 , 15 , 1 )

				E_term[ ipot ]   = 2 * E[ ipot ] / ( np.pi ** 2 )
				L_x_term[ ipot ] = 1 / ( L_x[ ipot ] ** 2 )

		L_y = 1 / ( np.sqrt( E_term - L_x_term ) )

		# Switches L_x and L_y elements with 50% probability.
		switch = np.random.default_rng( ).integers( 0 , 2 ,	num_pot )

		for ipot in range( 0 , num_pot ):
			if switch[ ipot ] == 1:
				L_x_temp 	= L_x[ ipot ]
				L_x[ ipot ] = L_y[ ipot ]
				L_y[ ipot ] = L_x_temp

		# Sets attributes.		
		self.c_x = c_x
		self.c_y = c_y
		self.L_x = L_x
		self.L_y = L_y

		# Turns potential list to stacked potential array.
		V_batch = np.array( iw_loop_gen( num_pot , image_size , x , y , c_x , c_y , L_x , L_y ) ).reshape( num_pot , image_size , image_size )

		# Sets attributes.
		self.V_batch = V_batch

		# Generates figures for each potential with 3D and 2D subplots.
		self.potential_plot_gen( plot , num_pot , x , y , V_batch )

		# Returns class instance.
		return( self )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: dig_potential_gen
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def dig_potential_gen( self , plot = False ):
		'''
			This function generates double-well inverted Gaussian potentials.

			Arguments:

				- self: Class instance.
				- plot: Show 3D and 2D plots of this potential of this type (dtype = bool).

			Returned values:

				- A_1: Well 1 depth (dtype = float).
				- A_2: Well 2 depth (dtype = float).
				- c_x_1: Well 1 center x coordinate (dtype = float).
				- c_y_1: Well 1 center y coordinate (dtype = float).
				- c_x_2: Well 2 center x coordinate (dtype = float).
				- c_y_2: Well 2 center y coordinate (dtype = float).
				- k_x_1: Well 1 x-axis width (dtype = float).
				- k_y_1: Well 1 y-axis width (dtype = float).
				- k_x_2: Well 2 x-axis width (dtype = float).
				- k_y_2: Well 2 y-axis width (dtype = float).
				- V_batch: num_pot double-well inverted Gaussian potentials (dtype = numpy.ndarray).
		'''

		# ---------------------------------------------------------------------------------------------------------------------------------------------
		# Function: dig_loop_gen
		# ---------------------------------------------------------------------------------------------------------------------------------------------

		@jit( nopython = True ) # Translates function to machine language in order to compile quickly. 
		def dig_loop_gen( num_pot , image_size , x , y , A_1 , A_2 , c_x_1 , c_y_1 , c_x_2 , c_y_2 , k_x_1 , k_y_1 , k_x_2 , k_y_2 ):
			'''
				This executes the potential batch generation loop for double-well inverted Gaussian potentials.

				Arguments:

					- image_size: Image dimensions (dtype = int).
					- num_pot: Number of potentials to generate (dtype = int).
					- x: x data in a.u. (dtype = numpy.ndarray).
					- y: y data in a.u. (dtype = numpy.ndarray).
					- A_1: Well 1 depth (dtype = float).
					- A_2: Well 2 depth (dtype = float).
					- c_x_1: Well 1 center x coordinate (dtype = float).
					- c_y_1: Well 1 center y coordinate (dtype = float).
					- c_x_2: Well 2 center x coordinate (dtype = float).
					- c_y_2: Well 2 center y coordinate (dtype = float).
					- k_x_1: Well 1 x-axis width (dtype = float).
					- k_y_1: Well 1 y-axis width (dtype = float).
					- k_x_2: Well 2 x-axis width (dtype = float).
					- k_y_2: Well 2 y-axis width (dtype = float).

				Returned values:

					- V_batch: num_pot simple harmonic oscillator potentials (dtype = numpy.ndarray).
			'''

			# -----------------------------------------------------------------------------------------------------------------------------------------
			# Function: dig_potential
			# -----------------------------------------------------------------------------------------------------------------------------------------

			def dig_potential( x , y , A_1 , A_2 , c_x_1 , c_y_1 , c_x_2 , c_y_2 , k_x_1 , k_y_1 , k_x_2 , k_y_2 ):
				'''
					This function generates the double-well inverted Gaussian potential.

					Arguments:

						- x: x data in a.u. (dtype = numpy.ndarray).
						- y: y data in a.u. (dtype = numpy.ndarray).
						- A_1: Well 1 depth (dtype = float).
						- A_2: Well 2 depth (dtype = float).
						- c_x_1: Well 1 center x coordinate (dtype = float).
						- c_y_1: Well 1 center y coordinate (dtype = float).
						- c_x_2: Well 2 center x coordinate (dtype = float).
						- c_y_2: Well 2 center y coordinate (dtype = float).
						- k_x_1: Well 1 x-axis width (dtype = float).
						- k_y_1: Well 1 y-axis width (dtype = float).
						- k_x_2: Well 2 x-axis width (dtype = float).
						- k_y_2: Well 2 y-axis width (dtype = float).

					Returned values:

						- V: The potential given the arguments (dtype = numpy.ndarray).
				'''

				# Well potentials.
				V_1 = - A_1 * np.exp( ( - ( ( ( x - c_x_1 ) / k_x_1 ) ** 2 ) ) + ( - ( ( ( y - c_y_1 ) / k_y_1 ) ** 2 ) ) )
				V_2 = - A_2 * np.exp( ( - ( ( ( x - c_x_2 ) / k_x_2 ) ** 2 ) ) + ( - ( ( ( y - c_y_2 ) / k_y_2 ) ** 2 ) ) )
				
				# Double-well inverted Gaussian potential function
				V = V_1 + V_2

				# Translation to zero minimum and scaling to 20 maximum.
				V = V + max( A_1 , A_2 )
				V = ( 20 / np.max( V ) ) * V

				return( V )

			# Generates potential list.
			V_batch = [ ]

			for ipot in range( 0 , num_pot ):
				V_batch.append( dig_potential( x , y , A_1[ ipot ] , A_2[ ipot ] , c_x_1[ ipot ] , c_y_1[ ipot ] , c_x_2[ ipot ] , 
											   c_y_2[ ipot ] , k_x_1[ ipot ] , k_y_1[ ipot ] , k_x_2[ ipot ] , k_y_2[ ipot ] ) )

			return( V_batch )

		# Stores attributes.
		num_pot    = self.num_pot
		image_size = self.image_size
		x 		   = self.x
		y 		   = self.y
        
		center = 8
		lower = 1.6
		upper = 7
		# Generates A_1, A_2, c_x_1, c_y_1, c_x_2, c_y_2, k_x_1, k_y_1, k_x_2, and k_y_2 arrays.
		A_1   = np.random.default_rng( ).uniform( 2 , 4 , num_pot )
		A_2   = np.random.default_rng( ).uniform( 2 , 4 , num_pot )
		c_x_1 = np.random.default_rng( ).uniform( - center , center , num_pot )
		c_y_1 = np.random.default_rng( ).uniform( - center , center , num_pot )
		c_x_2 = np.random.default_rng( ).uniform( - center , center , num_pot )
		c_y_2 = np.random.default_rng( ).uniform( - center , center , num_pot )
        
		for count in range(num_pot):
			distance = ((((c_x_1[count]) - c_x_2[count])**2 + (c_y_1[count] - c_y_2[count])**2) **(0.5))
			while  distance < 10:
				c_x_1[count] = np.random.default_rng( ).uniform( - center , center , 1 )[0]
				c_y_1[count] = np.random.default_rng( ).uniform( - center , center , 1 )[0]
				c_x_2[count] = np.random.default_rng( ).uniform( - center , center , 1 )[0]
				c_y_2[count] = np.random.default_rng( ).uniform( - center , center , 1 )[0]
				distance = ((((c_x_1[count]) - c_x_2[count])**2 + (c_y_1[count] - c_y_2[count])**2) ** (0.5))
                
		k_x_1 = np.random.default_rng( ).uniform( lower , upper , num_pot )
		k_y_1 = np.random.default_rng( ).uniform( lower , upper , num_pot )
		k_x_2 = np.random.default_rng( ).uniform( lower , upper , num_pot )
		k_y_2 = np.random.default_rng( ).uniform( lower , upper , num_pot )

		# Sets attributes.		
		self.A_1   = A_1
		self.A_2   = A_2
		self.c_x_1 = c_x_1
		self.c_y_1 = c_y_1
		self.c_x_2 = c_x_2
		self.c_y_2 = c_y_2
		self.k_x_1 = k_x_1
		self.k_y_1 = k_y_1
		self.k_x_2 = k_x_2
		self.k_y_2 = k_y_2

		# Turns potential list to stacked potential array.
		V_batch = np.array( dig_loop_gen( num_pot , image_size , x , y , A_1 , A_2 , c_x_1 , c_y_1 , c_x_2 , c_y_2 , k_x_1 , k_y_1 , k_x_2 , k_y_2 ) 
						  ).reshape( num_pot , image_size , image_size )

		# Sets attributes.
		self.V_batch = V_batch

		# Generates figures for each potential with 3D and 2D subplots.
		self.potential_plot_gen( plot , num_pot , x , y , V_batch )

		# Returns class instance.
		return( self )

	# -------------------------------------------------------------------------------------------------------------------------------------------------
	# Function: rnd_potential_gen
	# -------------------------------------------------------------------------------------------------------------------------------------------------

	def rnd_potential_gen( self , plot = False ):
		'''
			This function generates random potentials.

			Arguments:

				- self: Class instance.
				- plot: Show 3D and 2D plots of this potential of this type (dtype = bool).

			Returned values:

				- sigma_1: First Gaussian blur standard deviation floats (dtype = numpy.ndarray).
				- k: Random resolution integers (dtype = numpy.ndarray).
				- R: Random resolution integers (dtype = numpy.ndarray).
				- sigma_2: Second Gaussian blur standard deviation floats (dtype = numpy.ndarray).
				- d: Random exponentiation floats from list of values (dtype = numpy.ndarray).
				- V_batch: num_pot random potentials (dtype = numpy.ndarray).
		'''

		# @jit( nopython = True ) # Translates function to machine language in order to compile quickly. 
		def rnd_loop_gen( num_pot , image_size , x , y , sigma_1 , k , R , sigma_2 , d ):
			'''
				This executes the potential batch generation loop for random potentials.

				Arguments:

					- image_size: Image dimensions (dtype = int).
					- num_pot: Number of potentials to generate (dtype = int).
					- x: x data in a.u. (dtype = numpy.ndarray).
					- y: y data in a.u. (dtype = numpy.ndarray).
					- sigma_1: First Gaussian blur standard deviation floats (dtype = numpy.ndarray).
					- k: Random resolution integers (dtype = numpy.ndarray).
					- R: Random resolution integers (dtype = numpy.ndarray).
					- sigma_2: Second Gaussian blur standard deviation floats (dtype = numpy.ndarray).
					- d: Random exponentiation floats from list of values (dtype = numpy.ndarray).

				Returned values:

					- V_batch: num_pot simple harmonic oscillator potentials (dtype = numpy.ndarray).
			'''

			# -----------------------------------------------------------------------------------------------------------------------------------------
			# Function: rnd_potential
			# -----------------------------------------------------------------------------------------------------------------------------------------

			def rnd_potential( x , y , sigma_1 , k , R , sigma_2 , d , image_size ):
				'''
					This function generates the random potential.

					Arguments:

						- x: x data in a.u. (dtype = numpy.ndarray).
						- y: y data in a.u. (dtype = numpy.ndarray).
						- sigma_1: First Gaussian blur standard deviation floats (dtype = numpy.ndarray).
						- k: Random resolution integers (dtype = numpy.ndarray).
						- R: Random resolution integers (dtype = numpy.ndarray).
						- sigma_2: Second Gaussian blur standard deviation floats (dtype = numpy.ndarray).
						- d: Random exponentiation floats from list of values (dtype = numpy.ndarray).
						- image_size: Image dimensions (dtype = int).

					Returned values:

						- V: The potential given the arguments (dtype = numpy.ndarray).
				'''

				# Generate and resize first binary grid.
				grid_1 = np.array( np.random.default_rng( ).integers( 0 , 2 , [ 16 , 16 ] ) , dtype = "uint8" )
				grid_1 = np.array( resize( grid_1 , dsize = ( image_size , image_size ) ) , dtype = "float64" )

				# Generate and resize second binary grid.
				grid_2 = np.array( np.random.default_rng( ).integers( 0 , 2 , [ 16 , 16 ] ) , dtype = "uint8" )
				grid_2 = np.array( resize( grid_2 , dsize = ( int( image_size / 2 ) , int( image_size / 2 ) ) ) , dtype = "float64" )
				grid_2 = np.pad( grid_2 , int( image_size / 4 ) , mode = "constant" , constant_values = 0 )

				# Subtract the two grids, then blur and exponentiate results.
				grid = abs( grid_1 - grid_2 )
				grid = gaussian_filter( grid , sigma = sigma_1 )
				grid = grid ** d

				# Generate blob grid.
				blob = np.ones( [ image_size , image_size ] )

				# Randomly generate convex hull and return its path.
				blob_size 	= ( 25 / 32 ) * image_size
				blob_center = ( ( image_size - blob_size ) / 2 )
				blob_points = np.random.default_rng( ).integers( blob_center - 1 , blob_size + blob_center , [ k ** 2 , 2 ] )
				hull 	  	= ConvexHull( blob_points )
				hull_path 	= blob_points[ hull.vertices , : ]
				hull_path 	= np.vstack( [ hull_path , hull_path[ 0 , : ] ] )

				# Interpolate convex hull path.
				knots , u 				  = splprep( hull_path.T , k = 3 , per = 1 )
				hull_path_x , hull_path_y = splev( np.linspace( min( u ) , max( u ) , 1000 ) , knots , der = 0 )
				hull_path 				  = np.vstack( [ hull_path_x , hull_path_y ] ).T

				# Locate the convex hull path's centroid (center of mass).
				hull_center = Polygon( hull_path ).centroid.coords
				hull_center = hull_center[ 0 ]

				# Compute differences between centroid and grid center.
				hull_center_x = ( image_size / 2 ) - hull_center[ 0 ] - 1
				hull_center_y = ( image_size / 2 ) - hull_center[ 1 ] - 1

				# Recenter convec hull path to grid center.
				hull_path_x = hull_path[ : , 0 ] + hull_center_x
				hull_path_y = hull_path[ : , 1 ] + hull_center_y
				hull_path 	= np.vstack( [ hull_path_x , hull_path_y ] ).T
				hull_path 	= Path( hull_path )

				# Set blob grid points within convex hull path to 1 and those outside to 0, creating a binary mask.
				blob_indices = np.vstack( [ np.where( blob >= 0 )[ 0 ] , np.where( blob >= 0 )[ 1 ] ] ).T
				blob 		 = 1 * ( hull_path.contains_points( blob_indices ).reshape( image_size , image_size ) )
				blob 		 = blob.T

				# Resize blob to R x R resolution.
				blob = np.array( blob , dtype = "uint8" )
				blob = np.array( resize( blob , dsize = ( R , R ) ) , dtype = "float64" )

				# Pad blob edges with 0s for image_size x image_size resolution.
				if R % 2 == 0:
					blob = np.pad( blob , int( ( image_size - R ) / 2 ) , mode = "constant" , constant_values = 0 )
				elif R % 2 == 1:
					blob = np.pad( blob , int( ( image_size - R + 1 ) / 2 ) , mode = "constant" , constant_values = 0 )
					blob = blob[ : - 1 , : - 1 ]

				# Blur blob such that it smoothly goes from 1 in the interior to 0 at the edges.
				blob = gaussian_filter( blob , sigma = sigma_2 )

				# Multiplication of original grid and the blob mask, as well as inversion and scaling to 20 maximum.
				V = grid * blob
				V = ( np.max( V ) * np.ones( [ image_size , image_size ] ) ) - V
				V = ( 20 / np.max( V ) ) * V

				return( V )

			# Generates potential list.
			V_batch = [ ]

			for ipot in range( 0 , num_pot ):
				V_batch.append( rnd_potential( x , y , sigma_1[ ipot ] , k[ ipot ] , R[ ipot ] , sigma_2[ ipot ] , d[ ipot ] , image_size ) )

			return( V_batch )

		# Stores attributes.
		num_pot    = self.num_pot
		image_size = self.image_size
		x 		   = self.x
		y 		   = self.y

		# Generates sigma_1, k, R, sigma_2, and d arrays.
		sigma_1 = np.random.default_rng( ).uniform( ( 3 / 128 ) * image_size , ( 5 / 128 ) * image_size , num_pot )
		k 		= np.random.default_rng( ).integers( 2 , 8 , num_pot )
		R 		= np.random.default_rng( ).integers( ( 5 / 16 ) * image_size , ( ( 45 / 64 ) * image_size ) + 1 , num_pot )
		sigma_2 = np.random.default_rng( ).uniform( ( 5 / 128 ) * image_size , image_size / 16 , num_pot )
		d 		= np.random.default_rng( ).choice( [ 0.1 , 0.5 , 1.0 , 2.0 ] , num_pot )

		# Sets attributes.		
		self.sigma_1 = sigma_1
		self.k 		 = k
		self.R 		 = R
		self.sigma_2 = sigma_2
		self.d 		 = d

		# Turns potential list to stacked potential array.
		V_batch = np.array( rnd_loop_gen( num_pot , image_size , x , y , sigma_1 , k , R , sigma_2 , d ) ).reshape( num_pot , image_size , image_size )

		# Sets attributes.
		self.V_batch = V_batch

		# Generates figures for each potential with 3D and 2D subplots.
		self.potential_plot_gen( plot , num_pot , x , y , V_batch )

		# Returns class instance.
		return( self )


