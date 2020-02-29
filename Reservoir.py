import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.sparse
import sys

class Reservoir:
    def __init__(self, in_dim, res_dim, out_dim, timeConst=1, density=0.1, biasScale=1.0, SR = 1.0):
        """
        The initialization method for a Reservoir object.

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
        self.SR = SR
        self.output = 0

        self.inWeights = np.random.uniform(low=-1, high=1, size=(res_dim, in_dim))

        rvs = scipy.stats.norm().rvs
        W = scipy.sparse.random(m=res_dim, n=res_dim, density=density, data_rvs=rvs)
        W_rho = max(abs(np.linalg.eig(W.A)[0]))
        self.matrix = self.SR*W.A/W_rho

        self.bias = biasScale*np.random.uniform(low=-1, high=1, size=(res_dim))

        self.outWeights = np.zeros((out_dim, res_dim))
        self.times = np.zeros(1)
        self.state = np.zeros((1,res_dim))

    def train(self, targets, initTime = 0):
        """
        Trains the reservoir by changing self.outWeights

        Changes self.outWeights

        Args:
            inputs (matrix): The set of inputs that will be used for training
            targets (matrix): The set of targets training should aim for
        """
        tempTime = targets.shape[0]
        X = self.state[initTime:initTime + tempTime]
        #print(X)
        X_T = X.T
        self.outWeights = np.dot(np.dot(targets.T, X), np.linalg.inv(np.dot(X_T, X) + 1e-6*np.eye(self.res_dim)))
        def g_res(x, t):
            noise_g = 0
            return np.dot(self.outWeights, x.T).T + noise_g*np.random.normal(size=(self.out_dim))
        self.outputs = g_res(self.state, 0)

    def getVals(self, inputs, timeStep = .1):
        """
        For use once the reservoir is trained. Gives outputs based off the inputs
        """

        def f_res(x, u, t):
            c = 1
            #noise_f = 0
            if x.shape == (self.res_dim,):
                x = x.reshape(self.res_dim,1)
            #print(x.shape)
            temp1 = (-1/c)*x + (1/c)*np.tanh(np.dot(self.inWeights, u) + np.dot(self.matrix, x) + self.bias.reshape(50,1)) #+ noise_f*np.random.normal(size=(self.res_dim))
            #print(temp1.shape)
            return temp1

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

        propagations = runge_kutta_4(f_res, timeStep, self.state[-1].reshape(50,1), inputs, self.times[-1])
        propagations = propagations.reshape(1, self.res_dim)
        self.state = np.append(self.state, propagations, axis=0)
        self.output = g_res(propagations, 1)
        self.times = np.append(self.times, self.times[-1]+timeStep)
        return self.output

    def exportReservoir(self):
        """
        Exports the reservoir
        """
        #TODO figure out format of export
        #temp line to make IDE happy
        print("this will do something eventually")