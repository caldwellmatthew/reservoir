import Reservoir
import numpy as np

wanted = 50
inputM = np.array([[1]]).reshape(1,1)
outputM = np.array([[50]]).reshape(1,1)
output = 0

res = Reservoir.Reservoir(1, 50, 1)
res.train(outputM)
output = res.propagate(inputM)
outputM = np.append(outputM, [[50]], axis = 0)
print("input: ", inputM)
print("output: ", output)

for x in range(500):
    res.train(outputM)
    output = res.propagate(inputM)
    outputM = np.append(outputM, [[50]], axis = 0)

    print("input: ", inputM)
    print("output: ", output)

print(" ")
print("Done Training")
print(" ")

for x in range(50):
    #res.train(outputM)
    output = res.propagate(inputM)
    #outputM = np.append(outputM, [[50]], axis = 0)
    print("input: ", inputM)
    print("output: ", output)