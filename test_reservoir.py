from Reservoir import Reservoir
import math
import numpy as np
import matplotlib.pyplot as plt
import pytest

def train_n(res, num):
    inputM = np.array([[1]]).reshape(1,1)
    outputM = np.array([[num]]).reshape(1,1)
    output = 0

    for x in range(500):
        res.train(outputM)
        output = res.propagate(inputM)
        outputM = np.append(outputM, [[num]], axis=0)

    for x in range(50):
        output = res.propagate(inputM)

    return output

def train_sin(res):
    totalTime = int(math.pi * 100 * 8)
    linespace = np.linspace(0,1,totalTime)
    
    inputM = np.array(.5*math.sin(8*math.pi*linespace[0])).reshape(1,1)
    wantedOutputM = np.array(math.sin(8*math.pi*linespace[0])).reshape(1,1)
    res.train(wantedOutputM)
    res.propagate(np.array(.5*math.sin(8*math.pi*linespace[0])).reshape(1,1))

    #Training
    for x in linespace[1:]:
        inputM = np.append(inputM, np.array(.5*math.sin(8*math.pi*x)).reshape(1,1), axis = 0)
        wantedOutputM = np.append(wantedOutputM, np.array(math.sin(8*math.pi*x)).reshape(1,1), axis = 0)

        res.train(wantedOutputM)
        res.propagate(np.array(.5*math.sin(8*math.pi*x)).reshape(1,1))

    #Reset states
    res.resetState()

    #Get actual outputs
    output = res.propagate(np.array(inputM[0]).reshape(1,1))
    outputM = np.array(output).reshape(1,1)

    for x in inputM[1:]:
        output = res.propagate(np.array(x).reshape(1,1))
        outputM = np.append(outputM, np.array(output).reshape(1,1), axis = 0)

    # MSE = ((wantedOutputM - outputM)**2).mean(axis = None)

    # fig = plt.figure()
    # ax = fig.add_axes([0.1, 0.5, 0.8, 0.4], ylim=(-1.2, 1.6), ylabel='Displacement', xlabel='Time Steps')
    # ax.plot(inputM, 'b--', wantedOutputM, 'r')
    # ax.plot(outputM, 'g*', label = 'MSE = %f'  % MSE)
    # ax.legend(loc = "upper right")
    # plt.show()

    return outputM

def test_reset():
    '''Ensures resetting reservoir zeros states and times matrices.'''
    RES_DIM = 50
    res = Reservoir(1, RES_DIM, 1)
    train_n(res, 10)
    res.resetState()

    assert res.times.size == 1
    assert res.times[0] == 0

    assert res.state.size == RES_DIM
    for i in res.state[0]:
        assert i == 0

@pytest.fixture(params=list(range(10, 101, 10)))
def nv(request):
    return request.param

def test_n(nv):
    '''Test that the reservoir can obtain a numeric value.'''
    res = Reservoir(1, 50, 1)
    output = train_n(res, nv)
    assert abs(nv - output) < 0.1

sin_res = Reservoir(1, 50, 1)
sin_outputM = train_sin(sin_res)

def pi_index(mul=1):
    return int(mul * int(math.pi * 100) - 1)

def test_sin_pi():
    assert abs(math.sin(math.pi) - sin_outputM[pi_index()]) < 0.001

def test_sin_pi_over_2():
    assert abs(math.sin(math.pi / 2) - sin_outputM[pi_index(0.5)]) < 0.001

