# =============================================================================
# Imports
# =============================================================================
import os, sys, time
dir_path = os.path.dirname(__file__)
sys.path.append(str(dir_path))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from data_gen import potentials, potentials_gen
 
# =============================================================================
#  Class: Solver
# =============================================================================
class solver:
    '''
        Solve the TISE using the 2D Finite Difference Method via
        direct matrix.
    '''

    def __init__( self, lower_bound = -20 , upper_bound = 20,  grid_size = 128 ):

        self.grid_size = grid_size

        # Create Grid
        self.x = self.y = np.linspace(lower_bound, upper_bound, grid_size)

        # Grid Spacing
        self.dx = self.dy =  self.x[1] - self.x[0]

        # Grid Coordinates
        (self.x, self.y) = np.meshgrid(self.x, self.y, indexing="ij")

        # T and V are N**2 x N**2 matricies
        self.matrix_dim = ( self.grid_size**2, self.grid_size**2 )
        
        
        def build_hamiltonian(self):
            
            # Create KE Sparse matrix
            diagonal = sp.diags([-1,4,-1], offsets=[-1,0,1], shape=self.matrix_dim, format="csr" )
            superdiagonal = sp.diags( [-1], [self.grid_size], self.matrix_dim, format="csr" )
            subdiagonal = sp.diags( [-1], [-self.grid_size], self.matrix_dim, format="csr" )

    
            self.T_sparse = 0.5 * ( ( diagonal + superdiagonal + subdiagonal ) / (self.dx * self.dy) )

            # Create PE Sparse matrix
            self.V_sparse = sp.coo_matrix( self.matrix_dim )
        
            self.H = self.T_sparse 
    
            return self.H
        
        build_hamiltonian(self)        
        
    def solve(self, V):
        '''
            Compute ground state eigenenergy by diagonalizing
            the Hamiltonian matrix.

            Arguments:
                - V: Potential Energy Function (dtype = numpy.ndarray)
            
            Returned Values:
                - energy: Ground state energy of desired potential function (dtype = float)

        '''
        
        self.V_sparse.setdiag( V.flatten() )

        H_temp = self.V_sparse + self.H
        
        def eigen(H_temp):
            
            energy = eigs(H_temp, k=1, which="SR", return_eigenvectors=False)
            return energy
    
        energy = np.real(eigen(H_temp))

        return energy[0]
