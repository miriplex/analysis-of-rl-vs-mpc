# Inverted Pendulum Nominal Screening (nonlinear)

Training plant id: `inverted_pendulum_linearized`

| Variant | Pass? | Failure | Max |theta| [deg] | Max |x| [m] | Max |u| [N] | Final state norm |
| --- | :---: | --- | ---: | ---: | ---: | ---: |
| MPC | yes | -- | 5.000 | 0.129 | 5.269 | 0.000703 |
| history8 16_16 | yes | -- | 5.031 | 0.120 | 6.198 | 0.001354 |
| state6 32_32_32 | yes | -- | 5.000 | 0.137 | 5.336 | 0.001677 |
| rich11 32_32 | yes | -- | 5.000 | 0.125 | 5.708 | 0.002081 |
| history8 32_32_32 | yes | -- | 5.000 | 0.119 | 5.417 | 0.002102 |
| state6 16_16 | yes | -- | 5.000 | 0.110 | 5.858 | 0.002631 |
| rich11 16_16 | yes | -- | 5.000 | 0.115 | 5.638 | 0.003313 |
| history8 32_32 | yes | -- | 5.000 | 0.118 | 5.680 | 0.006670 |
| state6 32_32 | yes | -- | 5.000 | 0.128 | 5.446 | 0.007773 |
| rich11 32_32_32 | yes | -- | 5.424 | 0.345 | 7.285 | 0.247490 |
| PID-features | no | angle_limit | 91.850 | 0.185 | 11.113 | 7.795491 |
