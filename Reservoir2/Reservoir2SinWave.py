import numpy
import Reservoir2
import matplotlib.pyplot as plt
#Team Blue Angels: Eric Y

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
# Combine ax2 with Reservoir2 line 60 to 73 for the node values
# ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], ylabel='Displacement',
#                   ylim=(-1.2, 1.2), xlabel='Time Steps')
# Calling the functions to generate a reservoir
resin = Reservoir2.gresin(resDim, inputDim, 1)
res = Reservoir2.gres(spectralRadius, resDim, den)
resfb = Reservoir2.gresin(resDim, inputDim, inputDim)
resout = Reservoir2.trainRes(res, resin, resfb, resDim, outputDim, time, timeInput, timeOutput, washout, leakrate, noise, bias,
                             reg=1e-6)
resend = Reservoir2.runRes(res, resin, resfb, resout, resDim, outputDim, timeInput, leakrate, bias, time)
# Uncomment these lines to export the data from nodes to a yml file
# stores node data into yaml
# nx.write_yaml(res, 'test.yaml')
ax1.plot(timeInput, 'b--', timeOutput, 'r', resend, 'g*')
# Same as line 35
# ax2.legend(loc='upper right')
plt.show()