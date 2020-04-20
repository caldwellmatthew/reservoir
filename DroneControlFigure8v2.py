import logging
import time
import csv

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

from pandas import read_csv
from seaborn import relplot
import matplotlib.pyplot as plt

URI = 'radio://0/80/2M'
timeStamp = str(time.time())
logging.basicConfig(level=logging.ERROR)
PATH = '/home/bitcraze/Documents/Drone Flight/PID Untuned/' + timeStamp + '.csv'
PATH2 = '/home/bitcraze/Documents/Drone Flight/PID Untuned/checklist.csv'
maskRadius = .5
if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)

    lg_stab = LogConfig(name='stateEstimate', period_in_ms=10)
    lg_stab.add_variable('stateEstimate.x', 'float')
    lg_stab.add_variable('stateEstimate.y', 'float')

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)
    angle = 180  # degrees/s changes turn angle
    speed = .8  # m/s changes drone speed
    looptime = 360 // angle * 10
    # takeoff
    # send_hover_setpoint(vx(m/s),vy(m/s),yawrate(degrees/s),zdistance)

    cf.commander.send_hover_setpoint(0, 0, 0, 0.1)
    time.sleep(0.2)
    cf.commander.send_hover_setpoint(0, 0, 0, 0.2)
    time.sleep(0.15)
    cf.commander.send_hover_setpoint(0, 0, 0, 0.3)
    time.sleep(0.1)
    cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
    time.sleep(0.1)

    for _ in range(15):  # loop 2 seconds
        cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
        time.sleep(0.1)

    with SyncLogger(scf, lg_stab) as logger:
        for _ in range(int(looptime)):
            cf.commander.send_hover_setpoint(speed, 0, angle, 0.4)
        time.sleep(0.1)

        for _ in range(looptime):
            cf.commander.send_hover_setpoint(speed, 0, -angle, 0.4)
        time.sleep(0.1)

        logger._queue.put('DISCONNECT_EVENT')

        for _ in range(6):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
        time.sleep(0.1)

        # landing
        cf.commander.send_hover_setpoint(0, 0, 0, 0.14)
        time.sleep(0.1)
        cf.commander.send_hover_setpoint(0, 0, 0, 0.12)
        time.sleep(0.1)
        cf.commander.send_hover_setpoint(0, 0, 0, 0.10)
        time.sleep(0.1)
        cf.commander.send_hover_setpoint(0, 0, 0, 0.08)
        time.sleep(0.1)
        cf.commander.send_hover_setpoint(0, 0, 0, 0.06)
        time.sleep(0.1)
        cf.commander.send_hover_setpoint(0, 0, 0, 0.04)
        time.sleep(0.1)
        cf.commander.send_hover_setpoint(0, 0, 0, 0.02)
        time.sleep(0.1)
        cf.commander.send_hover_setpoint(0, 0, 0, 0.01)
        time.sleep(0.1)
        cf.commander.send_stop_setpoint()

        with open(PATH, mode='w') as file:
            writer = csv.writer(file, delimiter=',')
        writer.writerow(['time', 'X', 'Y'])
        failed = False
        first = True
        xOff = 0
        yOff = 0
        for entry in logger:
            if first:
                xOff = data["stateEstimate.x"]
            yOff = data["stateEstimate.y"]
            timestamp = entry[0]
            data = entry[1]
            logconf_name = entry[2]
            # print('[%d][%s]: %s' % (timestamp, logconf_name, data))
            point_x = data["stateEstimate.x"] - xOff
            point_y = data["stateEstimate.y"] - yOff
            ###Old Option
            # if (point_x > 0.6 or point_x < -0.3): # x out of bounds
            #	print("x out of bounds")
            #	failed = True
            # if (point_y > 0.5 or point_y < -0.6): # y out of bounds
            #	print("y out of bounds")
            #	failed = True
            # if (point_x > 0.4 and point_y < -0.2): # bottom right
            #	print("bottom right")
            #	failed = True
            # if (point_x < -0.1 and point_y > 0.2): # top left
            #	print("top left")
            #	failed = True
            ###New Option
            # (x-0)^2 + (y-.255)^2<Radius^2
            if point_x ** 2 + (point_y + .255) ** 2 > maskRadius ** 2 and point_x ** 2 + (
                    point_y - .255) ** 2 > maskRadius ** 2:
                failed = True
            print("Out of Bounds")
            writer.writerow([timestamp, point_x, point_y])
        if failed:
            print('Test failed: drone flight veered outside of defined mask.')
            with open(PATH2, mode='w') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(['Test', timeStamp, 'FAILED'])
        else:
            print('Test passed: drone flight remained within defined mask.')
            with open(PATH2, mode='w') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(['Test', timeStamp, 'PASSED'])

        flight = read_csv(PATH, header=0, names=['time', 'x', 'y'])
        relplot(x='x', y='y', data=flight)
        tcircle = plt.Circle((0, .255), maskRadius, color='g', alpha=.25, fill=True)
        bcircle = plt.Circle((0, -.255), maskRadius, color='g', alpha=.25, fill=True)
        plt.ylim(-.6, .6)
        plt.xlim(-.6, .6)
        ax = plt.gca()
        ax.add_artist(tcircle)
        ax.add_artist(bcircle)
        plt.show()

