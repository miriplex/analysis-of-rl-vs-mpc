# Inverted Pendulum Nominal Screening

Plant: `inverted_pendulum_linearized`

| Variant | Pass? | Failure | Max |theta| [deg] | Max |x| [m] | Max |u| [N] | Final state norm |
| --- | :---: | --- | ---: | ---: | ---: | ---: |
| history8 32_32_32 | yes | -- | 5.540 | 0.214 | 5.423 | 0.000932 |
| history8 32_32 | yes | -- | 5.145 | 0.163 | 5.073 | 0.005663 |
| history8 16_16 | yes | -- | 5.144 | 0.146 | 5.246 | 0.006178 |
| state6 32_32 | yes | -- | 5.117 | 0.155 | 4.973 | 0.006319 |
| state6 16_16 | yes | -- | 5.194 | 0.199 | 5.293 | 0.021023 |
| state6 32_32_32 | yes | -- | 5.109 | 0.189 | 4.842 | 0.035770 |
| MPC | yes | -- | 5.544 | 0.411 | 5.198 | 0.081833 |
| PID-features | no | angle_limit | 23.300 | 0.081 | 7.799 | 5.213015 |
