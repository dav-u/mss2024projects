import numpy as np
import matplotlib.pyplot as plt
import params


time = 0.0
x_ground_truth = np.array([
  params.START_POS,
  params.START_SPEED
])

x_history = []
z_history = []

while time < params.MAX_SIM_TIME:
  w = np.random.multivariate_normal(np.zeros(2), params.Q_ground_truth)

  x_ground_truth = params.F_ground_truth @ x_ground_truth + \
                   params.B_ground_truth @ params.u + w
  x_history.append(x_ground_truth)

  # simulate noisy measurement
  v = np.random.multivariate_normal(np.zeros(2), params.R_ground_truth)
  z = params.H_ground_truth @ x_ground_truth + v
  z_history.append(z)

  time += params.DT

x_history = np.array(x_history)
z_history = np.array(z_history)

plt.subplot(4, 1, 1)
plt.title("Position")
plt.ylabel("time [s]")
plt.ylabel("position [m]")
plt.plot(x_history[:, 0], label="True position")
plt.plot(z_history[:, 0], label="Measured position")
plt.legend()

plt.subplot(4, 1, 2)
plt.title("Speed")
plt.ylabel("time [s]")
plt.ylabel("speed [m/s]")
plt.plot(x_history[:, 1], label="True speed")
plt.plot(z_history[:, 1], label="Measured speed")
plt.legend()

plt.subplots_adjust(hspace=1)

plt.show()