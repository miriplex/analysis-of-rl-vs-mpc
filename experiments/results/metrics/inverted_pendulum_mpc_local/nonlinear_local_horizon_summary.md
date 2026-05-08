# Nonlinear Local MPC Horizon Sweep

Force bounds: `[-10.0, +10.0] N`
Angle survival limit: `20.0 deg`
Settlement rule: `|theta(T)| <= 2.0 deg`, `|x(T)| <= 0.10 m`, `||(x_dot, theta_dot)|| <= 0.50`

| Terminal cost | N | Passed / total | Survived / total | Pass frac | Survival frac | Max final |x| [m] | Max final |theta| [deg] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 20 | 0/18 | 18/18 | 0.000 | 1.000 | 1.565 | 0.012 |
| off | 30 | 0/18 | 18/18 | 0.000 | 1.000 | 1.422 | 0.019 |
| off | 40 | 0/18 | 18/18 | 0.000 | 1.000 | 1.417 | 0.020 |
| off | 60 | 0/18 | 18/18 | 0.000 | 1.000 | 1.417 | 0.020 |
| on | 20 | 0/18 | 18/18 | 0.000 | 1.000 | 1.133 | 0.315 |
| on | 30 | 0/18 | 18/18 | 0.000 | 1.000 | 1.423 | 0.020 |
| on | 40 | 0/18 | 18/18 | 0.000 | 1.000 | 1.417 | 0.020 |
| on | 60 | 0/18 | 18/18 | 0.000 | 1.000 | 1.417 | 0.020 |
