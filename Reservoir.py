import numpy as np
import scipy.stats
import scipy.sparse
import sys
import pathlib

class Reservoir:
    def __init__(self, in_dim, res_dim, out_dim, timeConst=1, density=0.1, biasScale=1.0, SR = 1.0, reg = 1e-6):
        """
        The initialization method for a Reservoir object.

        Args:
            in_dim (int): The amount of inputs to the reservoir
            res_dim (int): The n dimension for a n x n reservoir matrix
            out_dim (int): The amount of outputs for the reservoir
            timeConst (float): Time constant for the reservoir, may be useless
            density (float): A number between 0 and 1. The rate of connections in the reservoir matrix
            biasScale (float): Used to initialize the bias vector
            SR (float): Parameter that controls how out of control reservoir might go. Should be around 1
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

        self.inWeights = np.random.uniform(low=-1, high=1, size=(res_dim, in_dim))

        rvs = scipy.stats.norm().rvs
        W = scipy.sparse.random(m=res_dim, n=res_dim, density=density, data_rvs=rvs)
        W_rho = max(abs(np.linalg.eig(W.A)[0]))
        self.matrix = self.SR*W.A/W_rho

        self.bias = biasScale*np.random.uniform(low=-1, high=1, size=(res_dim, 1))

        self.outWeights = np.zeros((out_dim, res_dim))
        self.times = np.zeros(1)
        self.state = np.zeros((1,res_dim))

    def train(self, targets, initIndex = 0):
        """
        Trains the reservoir by changing self.outWeights

        Changes self.outWeights

        Args:
            targets (matrix): The set of targets the reservoir should aim for. Should be a matrix with
            a shape of (N, M) where N is the history of targets and M should be the size of out_dim
            initIndex (integer): the initial time index that will be trained on.
        """
        tempTime = targets.shape[0]
        X = self.state[initIndex:initIndex + tempTime]
        X_T = X.T
        self.outWeights = np.dot(np.dot(targets.T, X), np.linalg.inv(np.dot(X_T, X) + self.reg*np.eye(self.res_dim)))
        def g_res(x, t):
            noise_g = 0
            return np.dot(self.outWeights, x.T).T + noise_g*np.random.normal(size=(self.out_dim))
        self.outputs = g_res(self.state, 0)

    def propagate(self, inputs, timeStep = .1):
        """
        For use once the reservoir is trained. Gives outputs based off the inputs

        adds another row on self.state
        sets self.outputs to equal the new outputs
        adds the next time onto self.times
        returns self.outputs

        Args:
            inputs (matrix): An array of inputs to the reservoir of shape (N, 1) where N is of size
            in_dim
            timeStep (float): The time the reservoir will predict ahead by.
        """

        def f_res(x, u, t):
            noise_f = 0
            return (-1/self.timeConst)*x + (1/self.timeConst)*np.tanh(np.dot(self.inWeights, u) + np.dot(self.matrix, x) + self.bias) + noise_f*np.random.normal(size=(self.res_dim, 1))

        def g_res(x, t):
            noise_g = 0
            return np.dot(self.outWeights, x.T).T + noise_g*np.random.normal(size=(self.out_dim))

        def runge_kutta_4(f, h, x, u, t):
            """
            Function for solving a differential equation (with input) 
            according to a 4th-order Runge-Kutta method.

            Args:
                f (func): The differential equation to solve. It must take as
                arguments a state-space vector x of size (x_dim,), an input
                vector u of size (u_dim,) and a float time t. It must return
                a state-derivative vector xdot of size (x_dim,).
                h (float): The integration times-step.
                x (array): The current state-space vector.
                u (array): The current input vector.
                t (array): The current time.

            Returns:
                An array of size (x_dim,) that is the value of x at t + h,
                according to f and u.
            """

            def k1(f, h, x, u, t):
                return f(x, u, t)
            def k2(f, h, x, u, t):
                return f(x + (h/2)*k1(f, h, x, u, t), u, t + (h/2))
            def k3(f, h, x, u, t):
                return f(x + (h/2)*k2(f, h, x, u, t), u, t + (h/2))
            def k4(f, h, x, u, t):
                return f(x + h*k3(f, h, x, u, t), u, t + h)

            return x + (h/6)*(k1(f, h, x, u, t) + 2*k2(f, h, x, u, t) + 2*k3(f, h, x, u, t) + k4(f, h, x, u, t))

        propagations = runge_kutta_4(f_res, timeStep, self.state[-1].reshape(self.res_dim,1), inputs, self.times[-1])
        propagations = propagations.reshape(1, self.res_dim)
        self.state = np.append(self.state, propagations, axis=0)
        self.output = g_res(propagations, 1)
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
        Resets the states and times matrices
        """
        self.times = np.zeros(1)
        self.state = np.zeros((1,self.res_dim))
