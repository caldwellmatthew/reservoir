import numpy as np
import scipy.stats
import scipy.sparse
import sys
import pathlib

"""
This class is a general reservoir class implementing a general reservoir network. It is very customizable and can be easily tuned
by changing parameters. the reservoir is generated randomly every time the class is initialized.
Documentation in this class assumes the reader knows the basic theories of how reservoirs work.
See ReservoirTestSinWave for a good example of the use of this class.
"""
class Reservoir:
    def __init__(self, in_dim, res_dim, out_dim, timeConst=1, density=0.1, biasScale=1.0, SR = 1.0, reg = 1e-6):
        """
        The initialization method for a Reservoir object.
        If the size of the reservoir is low and the density is also low, you might end up with a reservoir without any connections.
        A reservoir without any connections won't work.

        Args:
            in_dim (int): The amount of inputs to the reservoir
            res_dim (int): The n dimension for a n x n reservoir matrix
            out_dim (int): The amount of outputs for the reservoir
            timeConst (float): Time constant for the reservoir, may be useless
            density (float): A number between 0 and 1. The rate of connections in the reservoir matrix.
            biasScale (float): Used to initialize the bias vector
            SR (float): The spectral radius of a reservoir. Determines how out of control reservoir might go. Should be around 1
            reg (float): The regularization parameter. The smaller reg is the more succeptable to noise the reservoir will be
        """
        self.in_dim = in_dim
        self.res_dim = res_dim
        self.out_dim = out_dim
        self.timeConst = timeConst
        self.density = density
        self.biasScale = biasScale
        self.SR = SR
        self.output = 0
        self.reg = reg

        # Calculate the input weights randomly
        self.inWeights = np.random.uniform(low=-1, high=1, size=(res_dim, in_dim))

        # Generate he random reservoir matrix
        rvs = scipy.stats.norm().rvs
        W = scipy.sparse.random(m=res_dim, n=res_dim, density=density, data_rvs=rvs)
        W_rho = max(abs(np.linalg.eig(W.A)[0]))
        self.matrix = self.SR*W.A/W_rho

        # Generate the random bias vector
        self.bias = biasScale*np.random.uniform(low=-1, high=1, size=(res_dim, 1))

        # initialize the output weights and the outputs to be all 0's
        self.outWeights = np.zeros((out_dim, res_dim))
        self.outputs = self.state = np.zeros((1,out_dim))

        # times is a growing record of when propogate was called.
        self.times = np.zeros(1)

        # state is a growing record of the outputs of the reservoir before they are multiplied by outWeights
        self.state = np.zeros((1,res_dim))

    def train(self, targets, initIndex = 0, outputNoise = 0):
        """
        Trains the reservoir by changing self.outWeights. Uses ridge regression to determine self.outWeights

        Changes self.outWeights and self.outputs

        Args:
            targets (matrix): The set of targets the reservoir should aim for. Should be a matrix with
            a shape of (N, M) where N is the history of targets and M should be the size of out_dim.
            initIndex (integer): the initial time index that will be trained on. Useful for when you want to train
            using data after some length of time.
            trainNoise (float): if not 0 will at some random noise to the outputs
        """
        tempTime = targets.shape[0]
        X = self.state[initIndex:initIndex + tempTime]
        X_T = X.T
        self.outWeights = np.dot(np.dot(targets.T, X), np.linalg.inv(np.dot(X_T, X) + self.reg*np.eye(self.res_dim)))
        self.outputs = self.getOutputs(self.state, outputNoise)

    def propagate(self, inputs, timeStep = .1, propNoise = 0, outputNoise = 0):
        """
        For use during training and then after training. takes inputs given to the reservoir and then calculates the next state and outputs of the reservoir.
        Noise can be added during training to improve learning. if the original training data has no noise, adding noise can be useful.

        adds another row on self.state
        sets self.outputs to equal the new outputs
        adds the next time onto self.times
        returns self.outputs

        Args:
            inputs (matrix): An array of inputs to the reservoir of shape (N, 1) where N is of size
            in_dim
            timeStep (float): The time the reservoir will predict ahead by.
            propNoise (float): if not 0 adds random noise during propagations
            outputNoise (float): if not 0 adds random noise to the outputs of the reservoir
        """
        # Uses runge kutta 4 method to estimate a hyperbolic tangent function. Used to get outputs from the reservoir
        nextState = self.runge_kutta_4(self.tanhFunction, timeStep, self.state[-1].reshape(self.res_dim,1), inputs, propNoise)
        nextState = nextState.reshape(1, self.res_dim)
        self.state = np.append(self.state, nextState, axis=0)
        self.output = self.getOutputs(nextState, outputNoise)
        self.times = np.append(self.times, self.times[-1]+timeStep)
        return self.output

    def exportReservoir(self, path = ""):
        """
        Exports the reservoir

        Args:
            path (string): useful for putting the data in a separate folder. Should follow format
            "Folder name here/"
        """
        tempPath = pathlib.Path(__file__).parent.resolve() / path 
        np.savetxt(tempPath / "reservoirWeights.txt", self.matrix, '%f', '\t')
        np.savetxt(tempPath / "inputWeights.txt", self.inWeights, '%f', '\t')
        np.savetxt(tempPath / "outputWeights.txt", self.outWeights, '%f', '\t')
        np.savetxt(tempPath / "states.txt", self.state, '%f', '\t')
        np.savetxt(tempPath / "times.txt", self.times, '%f', '\t')
        np.savetxt(tempPath / "bias.txt", self.bias, '%f', '\t')

    def resetState(self):
        """
        Resets the states and times matrices. Should be used in between training and actual use.
        """
        self.times = np.zeros(1)
        self.state = np.zeros((1,self.res_dim))

    def getOutputs(self, state, outputNoise = 0):
        """
        gets outputs by dot producting self.outweights by state

        Args:
            state (matrix): The state of the reservoir used to get outputs
            outputNoise (float): Amount of noise to be added to the outputs

        Returns: 
            An array of ouputs
        """
        return np.dot(self.outWeights, state.T).T + outputNoise*np.random.normal(size=(self.out_dim))

    def tanhFunction(self, state, inputs, propNoise = 0):
        """
        Hyperbolic tangent function used to get the next state of the reservoir.

        Args:
            state (array): The previous state of the reservoir
            inputs (array): The inputs to the reservoir
            propNoise (float): how much noise to add to the state

        Returns:
            the next state according to the hyperbolic tan function.
        """
        return (-1/self.timeConst)*state + (1/self.timeConst)*np.tanh(np.dot(self.inWeights, inputs) + np.dot(self.matrix, state) + self.bias) + propNoise*np.random.normal(size=(self.res_dim, 1))

    def runge_kutta_4(self, f, timeStep, state, inputs, propNoise = 0):
        """
        Function for solving a differential equation (with inputs) 
        according to a 4th-order Runge-Kutta method.

        Args:
            f (func): The differential equation to solve. It must take as
            arguments a state-space vector x of size (res_dim,), an input
            vector u of size (in_dim,). It must return
            a state-derivative vector xdot of size (res_dim,).
            timeStep (float): The integration times-step.
            state (array): The current state-space vector.
            inputs (array): The current input vector.
            propNoise (float): the amount of noise to be added to the calculated state

        Returns:
            An array of size (res_dim,) that is the next state according to the inputs.
        """

        def k1(f, timeStep, state, inputs):
            return f(state, inputs, propNoise)
        def k2(f, timeStep, x, inputs):
            return f(state + (timeStep/2)*k1(f, timeStep, state, inputs), inputs, propNoise)
        def k3(f, timeStep, state, inputs):
            return f(state + (timeStep/2)*k2(f, timeStep, state, inputs), inputs, propNoise)
        def k4(f, timeStep, state, inputs):
            return f(state + timeStep*k3(f, timeStep, state, inputs), inputs, propNoise)

        return state + (timeStep/6)*(k1(f, timeStep, state, inputs) + 2*k2(f, timeStep, state, inputs) + 2*k3(f, timeStep, state, inputs) + k4(f, timeStep, state, inputs))