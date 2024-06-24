# Kalman Filter

## Prediction step
$\^{x_k} = Fx_{k-1} + Bu_{k-1}$

$\^{P_k} = FP_{k-1}F^T + Q$

## Measurement step

$x_k = \^{x_k} + Ky$

$P_k = (I - KH)\^{P_k}$

with
$y = z_k - H \^{x_k}$

$S = H \^{P_k} H^T + R$

$K = \^{P_k} H^T S^{-1}$