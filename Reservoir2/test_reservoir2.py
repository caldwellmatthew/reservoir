import numpy
import Reservoir2
#Team Blue Angels: Eric Y


def test_gresin_not_none():
    res_dim = 1
    in_dim = 1
    input_scale = 1
    generated = Reservoir2.gresin(res_dim, in_dim, input_scale)

    assert len(generated) > 0


def test_gresin_inbounds():
    res_dim = 1
    in_dim = 1
    input_scale = 1
    generated = Reservoir2.gresin(res_dim, in_dim, input_scale)

    assert -1 < generated[0][0] < 1


def test_gresin_size():
    res_dim = 1
    in_dim = 1
    input_scale = 1
    generated = Reservoir2.gresin(res_dim, in_dim, input_scale)

    assert generated.shape == (1, 1)


def test_gresin_value():
    res_dim = 1
    in_dim = 1
    input_scale = 1
    generated = Reservoir2.gresin(res_dim, in_dim, input_scale)

    assert -1 < generated[0][0] < 1

def test_gres_not_none_len():
    SR = .8
    res_dim = 5
    density = .5
    generated = Reservoir2.gres(SR, res_dim, density)
    assert len(generated) > 0


def test_gres_not_none_value():
    SR = .8
    res_dim = 5
    density = .5
    generated = Reservoir2.gres(SR, res_dim, density)

    assert type(generated[0][0]) is numpy.double

def test_train_res():
    resDim = 30
    spectralRadius = .9
    den = .5
    inputDim = 1
    outputDim = 1
    time = 300
    leakrate = .1
    noise = .007
    washout = 50
    reg = 1e-6
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
    resin = Reservoir2.gresin(resDim, inputDim, 1)
    res = Reservoir2.gres(spectralRadius, resDim, den)
    resfb = Reservoir2.gresin(resDim, inputDim, inputDim)
    generated = Reservoir2.trainRes(res, resin, resfb, resDim, outputDim, time, timeInput, timeOutput, washout, leakrate, noise, bias, reg)
    assert len(generated) > 0

def test_run_res():
    resDim = 30
    spectralRadius = .9
    den = .5
    inputDim = 1
    outputDim = 1
    time = 300
    leakrate = .1
    noise = .007
    washout = 50
    reg = 1e-6
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
    resin = Reservoir2.gresin(resDim, inputDim, 1)
    res = Reservoir2.gres(spectralRadius, resDim, den)
    resfb = Reservoir2.gresin(resDim, inputDim, inputDim)
    resout = Reservoir2.trainRes(res, resin, resfb, resDim, outputDim, time, timeInput, timeOutput, washout,
                                 leakrate, noise, bias, reg)
    generated = Reservoir2.runRes(res, resin, resfb, resout, resDim, outputDim, timeInput, leakrate, bias, time)
    assert len(generated) > 0