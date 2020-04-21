import matplotlib.pyplot as plt
import numpy.linalg
import scipy.stats

"""Generates a sparsely connected matrix.
    SR = spectral radius
    res_dim = Dimensions of the reservoir N by N
    density = how strongly connected the nodes are
"""
def gres(SR, res_dim, density):
    # rvs is a randomly selected value generated per data point
    rvs = scipy.stats.norm().rvs
    W = scipy.sparse.random(m=res_dim, n=res_dim, density=density, data_rvs=rvs)
    W_rho = max(abs(numpy.linalg.eig(W.A)[0]))
    return SR * W.A / W_rho


"""Generates a input matrix that is used for training
    res_dim = Dimensions of the reservoir N by N
    in_dim = Input dimensions for the input values
    input_scale = Determines how strong the matrix is scaled by
"""
def gresin(res_dim, in_dim, input_scale):
    reSin = input_scale * numpy.random.uniform(low=-1, high=1, size=(res_dim, in_dim))
    return reSin


"""Trains the reservoir matrix with data.
    res = reservoir matrix that is sparsely connected.
    resin = reservoir input matrix that holds data for each iteration.
    resfb = reservoir feedback that is used to teach the data for each iteration
    resDim = the dimension of the reservoir input node matrix value
    outputDim = the dimension of the output node matrix value
    time = how many timesteps is used
    res_dim = Dimensions of the reservoir N by N
    timeInput = N by 1 matrix that holds all the input values
    timeOutput = N by 1 matrix that holds all the values that one wants
    washout = transient period to get rid of bad data, reservoir needs time to warm up
    leakrate = How fast the decay of each impulse is 
    noise = random value to knock the certainty of the value down to replicate a more real system
    bias = value to smooth out the data
    reg = regularization value which is alpha, usually really small
"""
def trainRes(res, resin, resfb, resDim, outputDim, time, timeInput, timeOutput, washout, leakrate, noise, bias, reg):
    M = numpy.zeros((time - washout, resDim))
    T = numpy.zeros((time - washout, outputDim))
    zeroMlen = numpy.random.uniform(low=0, high=0, size=(resDim, 1))
    zeroTlen = numpy.random.uniform(low=0, high=0, size=(1, 1))
    for t in range(0, time):
        u = timeInput[t]
        zeroMlen = (1 - leakrate) * zeroMlen + leakrate * numpy.tanh(
            numpy.dot(res, zeroMlen) + numpy.dot(resin, u) + numpy.dot(resfb, zeroTlen) + noise + bias)
        if t > washout:
            k = t - washout
            M[k] = numpy.transpose(zeroMlen)
            T[k] = timeOutput[t]
        zeroTlen = timeOutput[t]
    # Meant to plot out the possible values nodes in the matrix can hold
    node1 = []
    node2 = []
    node3 = []
    for x in range(0, len(M)):
        node1.append(M[x][0])
        node2.append(M[x][1])
        node3.append(M[x][2])
    ax2.plot(node1, 'y', label='node1')
    ax2.plot(node2, 'b', label='node2')
    ax2.plot(node3, 'g', label='node3')

    # Matrix multiplication to determine the reservoir values
    return numpy.dot(
        numpy.dot(numpy.transpose(T), M),
        numpy.linalg.inv(numpy.dot(numpy.transpose(M), M) + reg * numpy.eye(resDim)))

"""Runs the reservoir and through its possible states
    res = reservoir matrix that is sparsely connected.
    resin = reservoir input matrix that holds data for each iteration.
    resout = trained reservoir matrix that holds all the random values
    resfb = reservoir feedback that is used to teach the data for each iteration
    resDim = the dimension of the reservoir input node matrix value
    outputDim = the dimension of the output node matrix value
    timeInput = N by 1 matrix that holds all the input values
    leakrate = How fast the decay of each impulse is 
    bias = value to smooth out the data
    time = how many timesteps is used
"""
def runRes(res, resin, resfb, resout, resDim, outputDim, timeInput, leakrate, bias, time):
    statevalue = numpy.random.uniform(low=0, high=0, size=(resDim, 1))
    outputmatrix = numpy.random.uniform(low=0, high=0, size=(1, 1))
    states = numpy.zeros((time, resDim))
    outputs = numpy.zeros((time, outputDim))
    for t in range(0, time):
        u = timeInput[t]
        statevalue = (1 - leakrate) * statevalue + leakrate * numpy.tanh(
            numpy.dot(res, statevalue) + numpy.dot(resin, u) + numpy.dot(resfb, outputmatrix) + bias)
        outputmatrix = numpy.dot(resout, statevalue)
        states[t] = numpy.transpose(statevalue)
        outputs[t] = outputmatrix
    return outputs


# Meta/global Data points, used for parametrization of the functions
resDim = 30
spectralRadius = .9
den = .5
inputDim = 1
outputDim = 1
time = 300
leakrate = .1
noise = .007
washout = 50
# Input is training data values
timeInput = []
# New value that one wants to achieve with the reservoir
timeOutput = []
statematrix = numpy.zeros((time, resDim))
bias = numpy.random.uniform(low=-1, high=1, size=(resDim, 1))
# Initializing the test input and output data
# linspace returns evenly spaced numbers over a interval, this is need for time steps since
# sine waves work best with decimals
linespace = numpy.linspace(0, 1, time)
for x in linespace:
    timeInput.append(.5 * numpy.sin(8 * 3.14 * x))

for y in linespace:
    timeOutput.append(numpy.sin(8 * 3.14 * y))


fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], ylim=(-1.2, 1.2), ylabel='Displacement')
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], ylabel='Displacement',
                   ylim=(-1.2, 1.2), xlabel='Time Steps')
# Calling the functions to generate a reservoir
resin = gresin(resDim, inputDim, 1)
res = gres(spectralRadius, resDim, den)
resfb = gresin(resDim, inputDim, inputDim)
resout = trainRes(res, resin, resfb, resDim, outputDim, time, timeInput, timeOutput, washout, leakrate, noise, bias,
                  reg=1e-6)
resend = runRes(res, resin, resfb, resout, resDim, outputDim, timeInput, leakrate, bias, time)
# Uncomment these lines to export the data from nodes to a yml file
# stores node data into yaml
# nx.write_yaml(res, 'test.yaml')
ax1.plot(timeInput, 'b--', timeOutput, 'r', resend, 'g*')
ax2.legend(loc='upper right')
plt.show()
