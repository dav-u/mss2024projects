import numpy as np

class KalmanFilter:
  def __init__(
      self,
      initial_x: np.ndarray,
      initial_P: np.ndarray,
      F,
      B,
      Q,
      H,
      R):
    self.x = initial_x.copy()
    self.P = initial_P.copy()
    self.F = F 
    self.B = B
    self.Q = Q
    self.H = H
    self.R = R
    self.dim = len(self.x)

  def update(self, u, z):
    """
    Update the state and covariance matrix based on control input and measurements.
    u is control input.
    z are the measurements.
    """
    self.update_predict(u)
    self.update_measurement(z)
  
  def update_predict(self, u):
    self.x = self.F @ self.x + self.B @ u
    self.P = self.F @ self.P @ self.F.T + self.Q
  
  def update_measurement(self, z):
    self.y = z - (self.H @ self.x) # y is the "innovation": 0 -> measurements tell us nothing new
    self.S = self.H @ self.P @ self.H.T + self.R
    self.K = self.P @ self.H.T @ np.linalg.inv(self.S)

    self.x = self.x + self.K @ self.y
    self.P = (np.eye(self.dim) - self.K @ self.H) @ self.P