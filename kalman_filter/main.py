import numpy as np
import matplotlib.pyplot as plt
import params
from kalman_filter import KalmanFilter

time = 0.0
x_ground_truth = np.array([
  params.START_POS,
  params.START_SPEED
])

x_history = []
z_history = []

# estimated values
x_est_history = []

kalman_filter = KalmanFilter(
  initial_x=x_ground_truth.copy(),
  initial_P=np.array([[10.0, 0.0],
                      [0.0, 1.0]]),
  F=params.F_ground_truth,
  B=params.B_ground_truth,
  Q=params.Q_ground_truth,
  H=params.H_ground_truth,
  R=params.R_ground_truth,
)

while time < params.MAX_SIM_TIME:
  w = np.random.multivariate_normal(np.zeros(2), params.Q_ground_truth)

  x_ground_truth = params.F_ground_truth @ x_ground_truth + \
                   params.B_ground_truth @ params.u + w
  x_history.append(x_ground_truth)

  # simulate noisy measurement
  v = np.random.multivariate_normal(np.zeros(2), params.R_ground_truth)
  z = params.H_ground_truth @ x_ground_truth + v
  z_history.append(z)

  # update filter
  kalman_filter.update(params.u, z)
  x_est_history.append(kalman_filter.x.copy())

  time += params.DT

x_history = np.array(x_history)
z_history = np.array(z_history)
x_est_history = np.array(x_est_history)

kf_errors = x_history - x_est_history
meas_errors = x_history - z_history

print("KF mean abs error:", np.abs(kf_errors).mean())

plt.subplot(4, 1, 1)
plt.title("Position")
plt.ylabel("time [s]")
plt.ylabel("position [m]")
plt.plot(x_history[:, 0], label="True position", color="green")
plt.plot(z_history[:, 0], label="Measured position", color="red")
plt.plot(x_est_history[:, 0], label="Estimated position", color="blue")
plt.legend()

plt.subplot(4, 1, 2)
plt.title("Speed")
plt.ylabel("time [s]")
plt.ylabel("speed [m/s]")
plt.plot(x_history[:, 1], label="True speed", color="green")
plt.plot(z_history[:, 1], label="Measured speed", color="red")
plt.plot(x_est_history[:, 1], label="Estimated position", color="blue")
plt.legend()

plt.subplot(4, 1, 3)
plt.title("Position Error")
plt.ylabel("time [s]")
plt.ylabel("error [m]")
plt.plot(kf_errors[:, 0], label="KF position error", color="blue")
plt.plot(meas_errors[:, 0], label="Measurement position error", color="red")
plt.legend()

plt.subplot(4, 1, 4)
plt.title("Speed Error")
plt.ylabel("time [s]")
plt.ylabel("error [m]")
plt.plot(kf_errors[:, 1], label="KF speed error", color="blue")
plt.plot(meas_errors[:, 1], label="Measurement speed error", color="red")
plt.legend()

plt.subplots_adjust(hspace=1)

plt.show()