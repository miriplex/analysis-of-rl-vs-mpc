# Inverted pendulum measurement-noise stress test

Method:
- Use the nominally passing pendulum controllers from the nonlinear nominal screen.
- Re-screen them on the 18-case local nonlinear stabilization grid.
- Apply additive measurement noise to the observed state only.
- Use absorbing elimination: once a controller fails a stress level, it is not tested at higher levels.
- Base measurement-noise standard deviation: `[0.002, 0.02, 0.004363323129985824, 0.017453292519943295]` in `[x, x_dot, theta, theta_dot]` units.

## Nominal Local Screen

| Controller | Passed / total | Pass all? | Failed cases | Worst final state norm |
| --- | ---: | :---: | --- | ---: |
| MPC | 18/18 | yes | -- | 0.001177 |
| rich11 32_32 | 18/18 | yes | -- | 0.003480 |
| state6 16_16 | 18/18 | yes | -- | 0.005618 |
| rich11 16_16 | 18/18 | yes | -- | 0.005622 |
| state6 32_32_32 | 18/18 | yes | -- | 0.005829 |
| history8 16_16 | 18/18 | yes | -- | 0.006750 |
| history8 32_32_32 | 18/18 | yes | -- | 0.008944 |
| state6 32_32 | 18/18 | yes | -- | 0.011299 |
| history8 32_32 | 18/18 | yes | -- | 0.012995 |
| rich11 32_32_32 | 1/18 | no | x+0.00__theta+3.0,x+0.00__theta+8.0,x+0.00__theta-3.0,x+0.00__theta-5.0,x+0.00__theta-8.0,x+0.05__theta+3.0,x+0.05__theta+5.0,x+0.05__theta+8.0,x+0.05__theta-3.0,x+0.05__theta-5.0,x+0.05__theta-8.0,x-0.05__theta+3.0,x-0.05__theta+5.0,x-0.05__theta+8.0,x-0.05__theta-3.0,x-0.05__theta-5.0,x-0.05__theta-8.0 | 2.364879 |

## Controller Survival Summary

| Controller | Max fully passed stress_scale | First failure stress_scale | Failure reasons at first failure | Failed cases at first failure |
| --- | ---: | ---: | --- | --- |
| rich11 16_16 | 5.0 | 6.0 | final_rate_not_settled | x+0.00__theta+8.0,x-0.05__theta-8.0 |
| state6 32_32_32 | 4.0 | 5.0 | final_rate_not_settled,final_x_not_settled | x-0.05__theta+3.0,x-0.05__theta+8.0 |
| history8 32_32_32 | 4.0 | 5.0 | final_rate_not_settled,final_x_not_settled | x+0.00__theta+3.0,x+0.00__theta+5.0,x-0.05__theta+3.0 |
| history8 32_32 | 4.0 | 5.0 | final_rate_not_settled,final_x_not_settled | x+0.00__theta+3.0,x+0.00__theta+5.0,x-0.05__theta+3.0 |
| history8 16_16 | 4.0 | 5.0 | final_rate_not_settled | x+0.00__theta+3.0,x-0.05__theta+8.0 |
| state6 32_32 | 4.0 | 5.0 | final_rate_not_settled | x-0.05__theta+8.0 |
| rich11 32_32 | 4.0 | 5.0 | final_rate_not_settled | x-0.05__theta+8.0 |
| state6 16_16 | 4.0 | 5.0 | final_rate_not_settled | x-0.05__theta+8.0 |
| MPC | 3.0 | 4.0 | final_theta_not_settled | x+0.05__theta+3.0 |
| rich11 32_32_32 | 0.0 | -- | -- | -- |
