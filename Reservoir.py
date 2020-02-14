import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.sparse
import sys

class Reservoir:
    def __init__(self, in_dim, res_dim, out_dim, timeConst=1, density=0.1, biasScale=1.0):
        """The initialization method for a Reservoir object.

        Args:
            in_dim (int): The amount of inputs to the reservoir
            res_dim (int): The n dimension for a n x n reservoir matrix
            out_dim (int): The amount of outputs for the reservoir
            timeConst (float): Time constant for the reservoir, may be useless
            density (float): A number between 0 and 1. The rate of connections in the reservoir matrix
            biasScale (float): Used to initialize the bias vector
        """
        self.in_dim = in_dim
        self.res_dim = res_dim
        self.out_dim = out_dim
        self.timeConst = timeConst
        self.density = density
        self.biasScale = biasScale

        self.inWeights = np.random.uniform(low=-1, high=1, size=(res_dim, in_dim))

        rvs = scipy.stats.norm().rvs
        W = scipy.sparse.random(m=res_dim, n=res_dim, density=density, data_rvs=rvs)
        W_rho = max(abs(np.linalg.eig(W.A)[0]))
        self.matrix = W.A/W_rho

        self.bias = biasScale*np.random.uniform(low=-1, high=1, size=(res_dim))

        self.outWeights = np.zeros((out_dim, res_dim))

    def train(self, inputs, targets):
        """Trains the reservoir by changing self.outWeights

        Changes self.outWeights

        Args:
            inputs (matrix): The set of inputs that will be used for training
            targets (matrix): The set of targets training should aim for
        """
        inputsTransposed = inputs.T
        self.outWeights = np.dot(np.dot(targets.T, inputs), np.linalg.inv(np.dot(inputsTransposed, inputs) + 1e-6*np.eye(self.res_dim)))

    def getVals(self, inputs):
        """For use once the reservoir is trained. Gives outputs based off the inmputs
        """
        #TODO make getvals function 
        #temp line to make IDE happy
        return inputs * self.outWeights

    def exportReservoir(self):
        """Exports the reservoir
        """
        #TODO figure out format of export
        #temp line to make IDE happy
        print("this will do something eventually")