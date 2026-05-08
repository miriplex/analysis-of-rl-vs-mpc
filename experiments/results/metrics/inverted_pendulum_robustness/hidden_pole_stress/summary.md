# Inverted pendulum hidden-pole stress test

Method:
- Use the nominally passing pendulum controllers from the nonlinear nominal screen.
- Re-screen them on the 18-case local nonlinear stabilization grid.
- Add an unmodelled actuator lag chain `H(s) = (a / (s + a))^n` with `a = 100 rad/s` between controller output and plant input.
- Increase the hidden-pole order `n` while keeping the controller model unchanged.
- Use absorbing elimination: once a controller fails an order, it is not tested at higher orders.

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

| Controller | Max fully passed hidden_order | First failure hidden_order | Failure reasons at first failure | Failed cases at first failure |
| --- | ---: | ---: | --- | --- |
| MPC | 4.0 | 5.0 | final_rate_not_settled,final_theta_not_settled | x+0.00__theta+3.0,x+0.00__theta+5.0,x+0.00__theta+8.0,x+0.00__theta-3.0,x+0.00__theta-5.0,x+0.00__theta-8.0,x+0.05__theta+3.0,x+0.05__theta+5.0,x+0.05__theta+8.0,x+0.05__theta-3.0,x+0.05__theta-5.0,x+0.05__theta-8.0,x-0.05__theta+3.0,x-0.05__theta+5.0,x-0.05__theta+8.0,x-0.05__theta-3.0,x-0.05__theta-5.0,x-0.05__theta-8.0 |
| rich11 32_32 | 3.0 | 4.0 | final_rate_not_settled,final_theta_not_settled | x+0.00__theta+3.0,x+0.00__theta+5.0,x+0.00__theta+8.0,x+0.00__theta-3.0,x+0.00__theta-5.0,x+0.00__theta-8.0,x+0.05__theta+3.0,x+0.05__theta+5.0,x+0.05__theta+8.0,x+0.05__theta-3.0,x+0.05__theta-5.0,x+0.05__theta-8.0,x-0.05__theta+3.0,x-0.05__theta+5.0,x-0.05__theta+8.0,x-0.05__theta-3.0,x-0.05__theta-5.0,x-0.05__theta-8.0 |
| history8 32_32 | 3.0 | 4.0 | final_rate_not_settled,final_theta_not_settled | x+0.00__theta+3.0,x+0.00__theta+5.0,x+0.00__theta+8.0,x+0.00__theta-3.0,x+0.00__theta-5.0,x+0.00__theta-8.0,x+0.05__theta+3.0,x+0.05__theta+5.0,x+0.05__theta+8.0,x+0.05__theta-3.0,x+0.05__theta-5.0,x+0.05__theta-8.0,x-0.05__theta+3.0,x-0.05__theta+5.0,x-0.05__theta+8.0,x-0.05__theta-3.0,x-0.05__theta-5.0,x-0.05__theta-8.0 |
| rich11 16_16 | 3.0 | 4.0 | final_rate_not_settled,final_theta_not_settled | x+0.00__theta+5.0,x+0.00__theta+8.0,x+0.00__theta-8.0,x+0.05__theta+5.0,x+0.05__theta+8.0,x+0.05__theta-3.0,x+0.05__theta-5.0,x+0.05__theta-8.0,x-0.05__theta+3.0,x-0.05__theta+5.0,x-0.05__theta+8.0,x-0.05__theta-8.0 |
| history8 32_32_32 | 3.0 | 4.0 | angle_limit,final_rate_not_settled,final_theta_not_settled | x+0.00__theta+3.0,x+0.00__theta+5.0,x+0.00__theta+8.0,x+0.00__theta-3.0,x+0.00__theta-5.0,x+0.00__theta-8.0,x+0.05__theta+3.0,x+0.05__theta+5.0,x+0.05__theta+8.0,x+0.05__theta-3.0,x+0.05__theta-5.0,x+0.05__theta-8.0,x-0.05__theta+3.0,x-0.05__theta+5.0,x-0.05__theta+8.0,x-0.05__theta-3.0,x-0.05__theta-5.0,x-0.05__theta-8.0 |
| state6 32_32 | 2.0 | 3.0 | final_rate_not_settled | x+0.00__theta+3.0,x+0.00__theta+5.0,x+0.00__theta-3.0,x+0.00__theta-8.0,x+0.05__theta+3.0,x+0.05__theta+8.0,x+0.05__theta-3.0,x+0.05__theta-5.0,x-0.05__theta+3.0,x-0.05__theta+5.0,x-0.05__theta-3.0,x-0.05__theta-8.0 |
| state6 16_16 | 2.0 | 3.0 | final_rate_not_settled | x+0.00__theta+3.0,x+0.00__theta+5.0,x+0.00__theta+8.0,x+0.00__theta-3.0,x+0.00__theta-5.0,x+0.00__theta-8.0,x+0.05__theta+3.0,x+0.05__theta+5.0,x+0.05__theta+8.0,x+0.05__theta-3.0,x+0.05__theta-5.0,x+0.05__theta-8.0,x-0.05__theta+3.0,x-0.05__theta+5.0,x-0.05__theta+8.0,x-0.05__theta-3.0,x-0.05__theta-5.0,x-0.05__theta-8.0 |
| history8 16_16 | 2.0 | 3.0 | angle_limit,final_rate_not_settled | x+0.00__theta+3.0,x+0.00__theta+5.0,x+0.00__theta+8.0,x+0.00__theta-3.0,x+0.00__theta-5.0,x+0.00__theta-8.0,x+0.05__theta+3.0,x+0.05__theta+5.0,x+0.05__theta+8.0,x+0.05__theta-3.0,x+0.05__theta-5.0,x+0.05__theta-8.0,x-0.05__theta+3.0,x-0.05__theta+5.0,x-0.05__theta+8.0,x-0.05__theta-3.0,x-0.05__theta-5.0,x-0.05__theta-8.0 |
| state6 32_32_32 | 1.0 | 2.0 | final_rate_not_settled | x+0.00__theta+5.0,x+0.00__theta-5.0,x+0.05__theta-5.0,x-0.05__theta+5.0 |
| rich11 32_32_32 | 0.0 | -- | -- | -- |
