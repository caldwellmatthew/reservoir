import Reservoir
import numpy as np
import math
import matplotlib.pyplot as plt

totalTime = 300
inputM = []
wantedOutputM = []
outputM = []
linespace = np.linspace(0,1,totalTime)

res = Reservoir.Reservoir(1, 50, 1)

#Training
for x in linespace:
    if inputM == []:
        inputM = np.array(.5*math.sin(8*3.14*x)).reshape(1,1)
        wantedOutputM = np.array(math.sin(8*314*x)).reshape(1,1)
    else:
        inputM = np.append(inputM, np.array(.5*math.sin(8*3.14*x)).reshape(1,1), axis = 0)
        wantedOutputM = np.append(wantedOutputM, np.array(math.sin(8*3.14*x)).reshape(1,1), axis = 0)

    res.train(wantedOutputM)
    res.propagate(np.array(.5*math.sin(8*3.14*x)).reshape(1,1))

#Reset states
res.resetState()

#Get actual outputs
for x in inputM:
    output = res.propagate(np.array(x).reshape(1,1))
    if outputM == []:
        outputM = np.array(output).reshape(1,1)
    else:
        outputM = np.append(outputM, np.array(output).reshape(1,1), axis = 0)

MSE = ((wantedOutputM - outputM)**2).mean(axis = None)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.5, 0.8, 0.4], ylim=(-1.2, 1.6), ylabel='Displacement', xlabel='Time Steps')
ax.plot(inputM, 'b--', wantedOutputM, 'r')
ax.plot(outputM, 'g*', label = 'MSE = %f'  % MSE)
ax.legend(loc = "upper right")
plt.show()