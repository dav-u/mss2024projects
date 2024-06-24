import numpy as np

MAX_SIM_TIME = 60 # in seconds

START_POS = 0 # in meter
START_SPEED = 13.8 # meter per second

DT = 1.0


# Process parameters (simulating the real world)
#____________________________________________________________________________

# F_ground_truth is something that is true for the world.
# When implementing a Kalman Filter we are not really able to change this.

# x = (pos, speed)^T
# pos = pos + speed*dt
# speed = speed

# F_ground_truth * x = |1.0  dt| | pos | = |pos + dt*speed|
#                      |0.0 1.0| |speed|   |    speed     |
F_ground_truth = np.array([[1.0, DT],
                           [0.0, 1.0]])


# e.g. a car accelerates from 100km/h to 150km/h in one hour
# a [m/s^2] = delta_speed / delta_time = 50,000m / (3600^2)s^2 = 0.00385
u = np.array([0.00385])

# pos_change = 0
# speed_change = dt*acceleration

# B_ground_truth * u = |0 | |a| = | 0  |
#                      |dt|       |dt*a|
B_ground_truth = np.array([[0],
                           [DT]])

# process noise
Q_ground_truth = np.array([[1.0, 0.0],
                           [0.0, 0.01]])
                          
H_ground_truth = np.array([[0.98, 0.0],
                           [0.0, 1.02]])

R_ground_truth = np.array([[100.0, 0.0],
                           [0.0,       1.0]])