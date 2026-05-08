# Disturbance stress test

Method:
- Screen all controllers on the nominal first-order plant set.
- Only controllers with zero nominal failures are eligible for the disturbance stress sweep.
- Reuse the standard disturbed benchmark traces and scale all disturbance terms by a severity multiplier `x`.
- Increase `x = start, start + step, ...` until no eligible controller passes all plants.

Rollout horizon: `20 s`
Reference step: `-2`
Base measurement noise std: `0.02`
Base disturbance noise std: `0.02`
Base disturbance step magnitude: `0.1` at `t = 5 s`
Tested disturbance scales: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]`

A disturbance scale `x` applies to:
- measurement noise: `x * 0.02`
- disturbance noise: `x * 0.02`
- disturbance step amplitude: `x * 0.1`

## Nominal screening
| controller | pass_all_plants | pass_count | num_plants | failure_count | mean_cost | mean_tail_rms_error | failed_plants |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MPC | yes | 9 | 9 | 0 | 0.1211 | 4.952e-13 |  |
| 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1245 | 0.0033 |  |
| RL (PID-features) | yes | 9 | 9 | 0 | 0.4372 | 0.0057 |  |
| 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1306 | 0.0087 |  |
| 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1353 | 0.0163 |  |
| 32_32 (rich11) | yes | 9 | 9 | 0 | 0.1329 | 0.0270 |  |
| 16_16 (compact6) | no | 8 | 9 | 1 | 1.730e+09 | 2.752e+04 | unstable__rhp_zero |
| 16_16_16 (rich11) | no | 8 | 9 | 1 | 3.126e+10 | 1.170e+05 | unstable__rhp_zero |
| 32_32_32 (compact6) | no | 8 | 9 | 1 | 5.139e+10 | 1.500e+05 | unstable__rhp_zero |
| 16_16_16 (compact6) | no | 8 | 9 | 1 | 6.021e+10 | 1.624e+05 | unstable__rhp_zero |

## Survival summary
| controller | eligible_after_nominal | max_pass_all_scale | first_failure_scale | failed_plants_at_first_failure | failure_reasons_at_first_failure | worst_tail_rms_at_first_failure |
| --- | --- | --- | --- | --- | --- | --- |
| MPC | yes | 16.0000 | 17.0000 | unstable__rhp_zero | peak_error | 8.1645 |
| RL (PID-features) | yes | 7.0000 | 8.0000 | unstable__lhp_zero | late_escalation | 0.7640 |
| 32_32_32 (rich11) | yes | 0.5000 | 0.6000 | unstable__rhp_zero | late_escalation | 0.2130 |
| 16_16 (rich11) | yes | 0.3000 | 0.4000 | unstable__rhp_zero | late_escalation | 0.3112 |
| 32_32 (compact6) | yes | 0.3000 | 0.4000 | unstable__rhp_zero | late_escalation,peak_error | 1.270e+04 |
| 32_32 (rich11) | yes | 0 | 0.1000 | unstable__rhp_zero | late_escalation | 0.3577 |
| 16_16 (compact6) | no | 0 |  |  |  | 0 |
| 16_16_16 (compact6) | no | 0 |  |  |  | 0 |
| 16_16_16 (rich11) | no | 0 |  |  |  | 0 |
| 32_32_32 (compact6) | no | 0 |  |  |  | 0 |

## Scale-by-scale progression
| disturbance_scale | controller | pass_all_plants | pass_count | num_plants | failure_count | mean_cost | worst_tail_rms_error | worst_peak_error | failed_plants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | MPC | yes | 9 | 9 | 0 | 0.1211 | 2.909e-12 | 2.8659 |  |
| 0 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1245 | 0.0151 | 2.6624 |  |
| 0 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1306 | 0.0294 | 2.5852 |  |
| 0 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4372 | 0.0340 | 5.7110 |  |
| 0 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1353 | 0.0942 | 2.6640 |  |
| 0 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.1329 | 0.2170 | 2.6543 |  |
| 0.1000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1305 | 0.0224 | 2.5862 |  |
| 0.1000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1247 | 0.0272 | 2.6625 |  |
| 0.1000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4374 | 0.0393 | 5.7146 |  |
| 0.1000 | MPC | yes | 9 | 9 | 0 | 0.1214 | 0.0480 | 2.8700 |  |
| 0.1000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1480 | 0.2640 | 2.6683 |  |
| 0.1000 | 32_32 (rich11) | no | 8 | 9 | 1 | 0.1411 | 0.3577 | 2.6566 | unstable__rhp_zero |
| 0.2000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1309 | 0.0350 | 2.5872 |  |
| 0.2000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1251 | 0.0441 | 2.6627 |  |
| 0.2000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4377 | 0.0447 | 5.7182 |  |
| 0.2000 | MPC | yes | 9 | 9 | 0 | 0.1222 | 0.0961 | 2.8742 |  |
| 0.2000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1603 | 0.3727 | 2.6726 |  |
| 0.3000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4382 | 0.0502 | 5.7218 |  |
| 0.3000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1257 | 0.0664 | 2.6629 |  |
| 0.3000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1330 | 0.1262 | 2.5881 |  |
| 0.3000 | MPC | yes | 9 | 9 | 0 | 0.1235 | 0.1441 | 2.8784 |  |
| 0.3000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1502 | 0.2914 | 2.6769 |  |
| 0.4000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4388 | 0.0557 | 5.7255 |  |
| 0.4000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1266 | 0.0970 | 2.6630 |  |
| 0.4000 | MPC | yes | 9 | 9 | 0 | 0.1254 | 0.1921 | 2.8825 |  |
| 0.4000 | 16_16 (rich11) | no | 8 | 9 | 1 | 0.1411 | 0.3112 | 2.5891 | unstable__rhp_zero |
| 0.4000 | 32_32 (compact6) | no | 8 | 9 | 1 | 4.548e+06 | 1.270e+04 | 3.862e+04 | unstable__rhp_zero |
| 0.5000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4395 | 0.0612 | 5.7291 |  |
| 0.5000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1279 | 0.1405 | 2.6632 |  |
| 0.5000 | MPC | yes | 9 | 9 | 0 | 0.1279 | 0.2401 | 2.8867 |  |
| 0.6000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4404 | 0.0668 | 5.7327 |  |
| 0.6000 | MPC | yes | 9 | 9 | 0 | 0.1308 | 0.2882 | 2.8909 |  |
| 0.6000 | 32_32_32 (rich11) | no | 8 | 9 | 1 | 0.1302 | 0.2130 | 2.6633 | unstable__rhp_zero |
| 0.7000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4414 | 0.0724 | 5.7363 |  |
| 0.7000 | MPC | yes | 9 | 9 | 0 | 0.1344 | 0.3362 | 2.8950 |  |
| 0.8000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4425 | 0.0780 | 5.7399 |  |
| 0.8000 | MPC | yes | 9 | 9 | 0 | 0.1384 | 0.3842 | 2.8992 |  |
| 0.9000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4437 | 0.0860 | 5.7435 |  |
| 0.9000 | MPC | yes | 9 | 9 | 0 | 0.1430 | 0.4322 | 2.9034 |  |
| 1.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4451 | 0.0955 | 5.7472 |  |
| 1.0000 | MPC | yes | 9 | 9 | 0 | 0.1482 | 0.4803 | 2.9075 |  |
| 1.1000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4466 | 0.1051 | 5.7508 |  |
| 1.1000 | MPC | yes | 9 | 9 | 0 | 0.1539 | 0.5283 | 2.9117 |  |
| 1.2000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4482 | 0.1146 | 5.7544 |  |
| 1.2000 | MPC | yes | 9 | 9 | 0 | 0.1601 | 0.5763 | 2.9159 |  |
| 1.3000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4500 | 0.1242 | 5.7580 |  |
| 1.3000 | MPC | yes | 9 | 9 | 0 | 0.1669 | 0.6243 | 2.9200 |  |
| 1.4000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4519 | 0.1337 | 5.7616 |  |
| 1.4000 | MPC | yes | 9 | 9 | 0 | 0.1742 | 0.6724 | 2.9242 |  |
| 1.5000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4539 | 0.1433 | 5.7652 |  |
| 1.5000 | MPC | yes | 9 | 9 | 0 | 0.1821 | 0.7204 | 2.9284 |  |
| 1.6000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4561 | 0.1528 | 5.7689 |  |
| 1.6000 | MPC | yes | 9 | 9 | 0 | 0.1905 | 0.7684 | 2.9325 |  |
| 1.7000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4584 | 0.1624 | 5.7725 |  |
| 1.7000 | MPC | yes | 9 | 9 | 0 | 0.1994 | 0.8165 | 2.9367 |  |
| 1.8000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4608 | 0.1719 | 5.7761 |  |
| 1.8000 | MPC | yes | 9 | 9 | 0 | 0.2089 | 0.8645 | 2.9409 |  |
| 1.9000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4633 | 0.1815 | 5.7797 |  |
| 1.9000 | MPC | yes | 9 | 9 | 0 | 0.2190 | 0.9125 | 2.9450 |  |
| 2.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4660 | 0.1910 | 5.7833 |  |
| 2.0000 | MPC | yes | 9 | 9 | 0 | 0.2295 | 0.9605 | 2.9492 |  |
| 3.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4999 | 0.2865 | 5.8195 |  |
| 3.0000 | MPC | yes | 9 | 9 | 0 | 0.3651 | 1.4408 | 2.9909 |  |
| 4.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.5469 | 0.3820 | 5.8557 |  |
| 4.0000 | MPC | yes | 9 | 9 | 0 | 0.5549 | 1.9211 | 3.0326 |  |
| 5.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.6069 | 0.4775 | 5.8918 |  |
| 5.0000 | MPC | yes | 9 | 9 | 0 | 0.7990 | 2.4013 | 3.0742 |  |
| 6.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.6799 | 0.5730 | 5.9280 |  |
| 6.0000 | MPC | yes | 9 | 9 | 0 | 1.0973 | 2.8816 | 3.6352 |  |
| 7.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.7659 | 0.6685 | 5.9642 |  |
| 7.0000 | MPC | yes | 9 | 9 | 0 | 1.4498 | 3.3619 | 4.2411 |  |
| 8.0000 | MPC | yes | 9 | 9 | 0 | 1.8565 | 3.8421 | 4.8469 |  |
| 8.0000 | RL (PID-features) | no | 8 | 9 | 1 | 0.8650 | 0.7640 | 6.0003 | unstable__lhp_zero |
| 9.0000 | MPC | yes | 9 | 9 | 0 | 2.3175 | 4.3224 | 5.4528 |  |
| 10.0000 | MPC | yes | 9 | 9 | 0 | 2.8327 | 4.8026 | 6.0587 |  |
| 11.0000 | MPC | yes | 9 | 9 | 0 | 3.4022 | 5.2829 | 6.6645 |  |
| 12.0000 | MPC | yes | 9 | 9 | 0 | 4.0259 | 5.7632 | 7.2704 |  |
| 13.0000 | MPC | yes | 9 | 9 | 0 | 4.7038 | 6.2434 | 7.8763 |  |
| 14.0000 | MPC | yes | 9 | 9 | 0 | 5.4359 | 6.7237 | 8.4822 |  |
| 15.0000 | MPC | yes | 9 | 9 | 0 | 6.2223 | 7.2040 | 9.0880 |  |
| 16.0000 | MPC | yes | 9 | 9 | 0 | 7.0629 | 7.6842 | 9.6939 |  |
| 17.0000 | MPC | no | 8 | 9 | 1 | 7.9577 | 8.1645 | 10.2998 | unstable__rhp_zero |

## Failure breakdown
| scenario | controller | failure_count | failed_plants | failure_reasons | failure_details | late_escalation_plants |
| --- | --- | --- | --- | --- | --- | --- |
| disturbance_scale_0.1x | 32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.3577>0.2 and tail_g=1.697>1.2) | unstable__rhp_zero |
| disturbance_scale_0.4x | 16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.3112>0.2 and tail_g=2.671>1.2) | unstable__rhp_zero |
| disturbance_scale_0.4x | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=3.862e+04>10; tail_rms=1.27e+04>0.2 and tail_g=8.158>1.2) | unstable__rhp_zero |
| disturbance_scale_0.6x | 32_32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.213>0.2 and tail_g=1.74>1.2) | unstable__rhp_zero |
| disturbance_scale_17x | MPC | 1 | unstable__rhp_zero | peak_error | unstable__rhp_zero (peak_error=10.3>10) |  |
| disturbance_scale_8x | RL (PID-features) | 1 | unstable__lhp_zero | late_escalation | unstable__lhp_zero (tail_rms=0.2245>0.2 and tail_g=1.322>1.2) | unstable__lhp_zero |
| nominal_screen | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=7.531e+05>10; tail_rms=2.477e+05>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.444e+06>10; tail_rms=1.461e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=3.202e+06>10; tail_rms=1.053e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.105e+06>10; tail_rms=1.35e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |