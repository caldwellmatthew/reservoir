# Reservoir2.py
A generic library for creating, training, and running a reservoir network.
# ReservoirSinWaveSample.py
###Running the Code
If not using a IDE: Navigate to the correct folder and in the commandline type 'python3 Reservoir2SinWave.py' to run the program to 
generate a matlab plot from the reservoir.

If using something like Pycharm: Edit configuration and set the script path to the actual file. Then run in the IDE.


###Background
Creates a reservoir, trains it to create a sin wave, then outputs that sin wave.
Important parameters to make note of:

resDim: This will give the reservoir its dimensions. So resDim = 2 will create a 2 x 2 matrix, 
which is equivalent to 2 nodes.

spectralRadius: The larger spectral radius implies slower decay of impulse response and increase in an 
range of interactions

den: The amount of values in the matrix and connections. The higher density the more values with connections 
there are in the matrix.

inputDim = The value of the input reservoir value parameter

outputDim = The value of the output reservoir value.

time: Amount of timesteps used.

leakrate: The speed at which the input effects the reservoir.

noise: The value added to output values to stabilize the training data so it doesn't over train.

washout: Transient period, the reservoir needs time to 'warm up'

# test_reservoir.py
A unit test program for Reservoir2.py used to test different definitions. This checks if the code is doing what 
it's supposed to.
#Credits
Team Blue Angels of OSU CSE5911