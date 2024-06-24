import numpy as np

MAX_SIM_TIME = 60 # in seconds

START_POS = 0 # in meter
START_SPEED = 13.8 # meter per second

DT = 1.0

# This is something that is true for the world.
# When implementing a Kalman Filter we are not really able to change this.

# x = (pos, speed)^T
# pos = pos + speed*dt
# speed = speed
F_ground_truth = np.array([[1.0, DT],
                           [0.0, 1.0]])