# Kalman Filter

## Prediction step
$\hat{x_k} = Fx_{k-1} + Bu_{k-1}$

$\hat{P_k} = FP_{k-1}F^T + Q$

## Measurement step

$x_k = \hat{x_k} + Ky$

$P_k = (I - KH)\hat{P_k}$

with
$y = z_k - H \hat{x_k}$

$S = H \hat{P_k} H^T + R$

$K = \hat{P_k} H^T S^{-1}$