# Nonlinear Local MPC: Fixed vs Successive Linearization

Force bounds: `[-10.0, +10.0] N`
Angle survival limit: `20.0 deg`
Settlement rule: `|theta(T)| <= 2.0 deg`, `|x(T)| <= 0.10 m`, `||(x_dot, theta_dot)|| <= 0.50`

| Mode | N | Passed / total | Survived / total | Pass frac | Survival frac | Max final |x| [m] | Max final |theta| [deg] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 20 | 0/18 | 18/18 | 0.000 | 1.000 | 1.565 | 0.012 |
| fixed | 30 | 0/18 | 18/18 | 0.000 | 1.000 | 1.422 | 0.019 |
| fixed | 40 | 0/18 | 18/18 | 0.000 | 1.000 | 1.417 | 0.020 |
| fixed | 60 | 0/18 | 18/18 | 0.000 | 1.000 | 1.417 | 0.020 |
| successive | 20 | 0/18 | 18/18 | 0.000 | 1.000 | 1.982 | 0.053 |
| successive | 30 | 0/18 | 18/18 | 0.000 | 1.000 | 1.612 | 0.018 |
| successive | 40 | 0/18 | 18/18 | 0.000 | 1.000 | 1.589 | 0.022 |
| successive | 60 | 0/18 | 18/18 | 0.000 | 1.000 | 1.588 | 0.023 |
