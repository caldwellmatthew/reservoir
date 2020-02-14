import Reservoir
import numpy as np

def factoryFunc(temp):
    return temp * 100

wanted = 150
wantedOutput = np.array([[150]])
for x in range(49):
    wantedOutput = np.append(wantedOutput, [[150]])
input = 1
output = 0

for x in range(10):
    output = factoryFunc(input)
    print("input: ", input)
    print("output: ", output)
    if output > wanted:
        input = 1
    else :
        input = 2
res = Reservoir.Reservoir(1, 50, 1)
inputMatrix = np.array([[1, 2]])
for x in range(24):
    inputMatrix = np.append(inputMatrix, [[1, 2]])
input = 1
res.train(inputMatrix, wantedOutput)
print(res.outWeights)
"""
for x in range(10):
    output = res.getVals(input)
    print("input: ", input)
    print("output: ", output)
    if output > wanted:
        input = 1
    else :
        input = 2
"""