import Reservoir
import numpy as np

def factoryFunc(temp):
    return temp * 100

wanted = 150
wantedOutput = np.array([[50]])
currentOutputs = np.array([[100]])
inputMatrix = np.array([[1]])
for x in range(49):
    wantedOutput = np.append(wantedOutput, [[50]])
    currentOutputs = np.append(currentOutputs, [[100]])
    inputMatrix = np.append(inputMatrix, [[1]])
input = 1
inputM = np.array([[1]])
outputM = np.array([[50]])
outputM = outputM.reshape(1,1)
output = 0

wantedOutput = wantedOutput.reshape(50, 1)
currentOutputs = currentOutputs.reshape(50, 1)
inputMatrix = inputMatrix.reshape(50, 1)
inputMatrix = inputMatrix.T
wantedOutput = wantedOutput.T


for x in range(10):
    output = factoryFunc(input)
    print("input: ", input)
    print("output: ", output)
    if output > wanted:
        input = 1
    else :
        input = 2
res = Reservoir.Reservoir(1, 50, 1)
res.train(inputMatrix, outputM)
#output = res.getVals(inputM, currentOutputs)
for x in range(10):
    output = res.getVals(inputM, currentOutputs)
    currentOutputs = np.array(output)
    for x in range(49):
        currentOutputs = np.append(currentOutputs, output)
    currentOutputs = currentOutputs.reshape(50, 1)
    print("input: ", input)
    print("output: ", output[0][0])
    if (output[0][0] + 100) != wanted:
        temp = 50 - output[0][0]
        temp = np.array(temp)
        temp = temp.reshape(1, 1)
        res.train(inputMatrix, outputM)
        print(res.outWeights)
    else :
        break