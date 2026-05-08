# Step-disturbance stress test

Method:
- Screen all controllers on the nominal first-order plant set.
- Only controllers with zero nominal failures are eligible for the step-disturbance stress sweep.
- Increase the disturbance step amplitude by a multiplier `x` while keeping measurement noise and input disturbance noise at zero.
- Base disturbance step magnitude: `0.1` at `t = 5 s`

Rollout horizon: `20 s`
Reference step: `-2`
Tested stress scales: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]`

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
| MPC | yes | 20.0000 |  |  |  | 0 |
| RL (PID-features) | yes | 20.0000 |  |  |  | 0 |
| 32_32_32 (rich11) | yes | 1.0000 | 1.1000 | unstable__rhp_zero | late_escalation | 0.4784 |
| 32_32 (compact6) | yes | 0.8000 | 0.9000 | unstable__rhp_zero | late_escalation | 0.6263 |
| 32_32 (rich11) | yes | 0.5000 | 0.6000 | unstable__rhp_zero | late_escalation | 0.5151 |
| 16_16 (rich11) | yes | 0.3000 | 0.4000 | unstable__rhp_zero | late_escalation | 0.3423 |
| 16_16 (compact6) | no | 0 |  |  |  | 0 |
| 16_16_16 (compact6) | no | 0 |  |  |  | 0 |
| 16_16_16 (rich11) | no | 0 |  |  |  | 0 |
| 32_32_32 (compact6) | no | 0 |  |  |  | 0 |

## Scale-by-scale progression
| stress_scale | controller | pass_all_plants | pass_count | num_plants | failure_count | mean_cost | worst_tail_rms_error | worst_peak_error | failed_plants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | MPC | yes | 9 | 9 | 0 | 0.1211 | 2.909e-12 | 2.8659 |  |
| 0 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1245 | 0.0151 | 2.6624 |  |
| 0 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1306 | 0.0294 | 2.5852 |  |
| 0 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4372 | 0.0340 | 5.7110 |  |
| 0 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1353 | 0.0942 | 2.6640 |  |
| 0 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.1329 | 0.2170 | 2.6543 |  |
| 0.1000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1305 | 0.0224 | 2.5852 |  |
| 0.1000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1247 | 0.0270 | 2.6624 |  |
| 0.1000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4373 | 0.0394 | 5.7110 |  |
| 0.1000 | MPC | yes | 9 | 9 | 0 | 0.1213 | 0.0479 | 2.8659 |  |
| 0.1000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1358 | 0.1103 | 2.6640 |  |
| 0.1000 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.1373 | 0.2289 | 2.6543 |  |
| 0.2000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1308 | 0.0264 | 2.5852 |  |
| 0.2000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1250 | 0.0427 | 2.6624 |  |
| 0.2000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4374 | 0.0447 | 5.7110 |  |
| 0.2000 | MPC | yes | 9 | 9 | 0 | 0.1220 | 0.0958 | 2.8659 |  |
| 0.2000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1366 | 0.1319 | 2.6640 |  |
| 0.2000 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.1401 | 0.3088 | 2.6543 |  |
| 0.3000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4376 | 0.0501 | 5.7110 |  |
| 0.3000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1255 | 0.0637 | 2.6624 |  |
| 0.3000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1324 | 0.0977 | 2.5852 |  |
| 0.3000 | MPC | yes | 9 | 9 | 0 | 0.1232 | 0.1436 | 2.8659 |  |
| 0.3000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1379 | 0.1590 | 2.6640 |  |
| 0.3000 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.1429 | 0.3695 | 2.6543 |  |
| 0.4000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4379 | 0.0554 | 5.7110 |  |
| 0.4000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1262 | 0.0925 | 2.6624 |  |
| 0.4000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1396 | 0.1911 | 2.6640 |  |
| 0.4000 | MPC | yes | 9 | 9 | 0 | 0.1249 | 0.1915 | 2.8659 |  |
| 0.4000 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.1459 | 0.4243 | 2.6543 |  |
| 0.4000 | 16_16 (rich11) | no | 8 | 9 | 1 | 0.1396 | 0.3423 | 2.5852 | unstable__rhp_zero |
| 0.5000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4383 | 0.0608 | 5.7110 |  |
| 0.5000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1272 | 0.1329 | 2.6624 |  |
| 0.5000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1417 | 0.2277 | 2.6640 |  |
| 0.5000 | MPC | yes | 9 | 9 | 0 | 0.1271 | 0.2394 | 2.8659 |  |
| 0.5000 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.1502 | 0.4828 | 2.6543 |  |
| 0.6000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4387 | 0.0661 | 5.7110 |  |
| 0.6000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1288 | 0.1881 | 2.6624 |  |
| 0.6000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1445 | 0.2702 | 2.6640 |  |
| 0.6000 | MPC | yes | 9 | 9 | 0 | 0.1297 | 0.2873 | 2.8659 |  |
| 0.6000 | 32_32 (rich11) | no | 8 | 9 | 1 | 0.1553 | 0.5151 | 2.6543 | unstable__rhp_zero |
| 0.7000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4391 | 0.0715 | 5.7110 |  |
| 0.7000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1310 | 0.2522 | 2.6624 |  |
| 0.7000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1485 | 0.3270 | 2.6640 |  |
| 0.7000 | MPC | yes | 9 | 9 | 0 | 0.1328 | 0.3352 | 2.8659 |  |
| 0.8000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4397 | 0.0768 | 5.7110 |  |
| 0.8000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1338 | 0.2998 | 2.6624 |  |
| 0.8000 | MPC | yes | 9 | 9 | 0 | 0.1364 | 0.3830 | 2.8659 |  |
| 0.8000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1556 | 0.4273 | 2.6640 |  |
| 0.9000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4403 | 0.0822 | 5.7110 |  |
| 0.9000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1367 | 0.2980 | 2.6624 |  |
| 0.9000 | MPC | yes | 9 | 9 | 0 | 0.1404 | 0.4309 | 2.8659 |  |
| 0.9000 | 32_32 (compact6) | no | 8 | 9 | 1 | 0.1699 | 0.6263 | 2.6640 | unstable__rhp_zero |
| 1.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4409 | 0.0875 | 5.7110 |  |
| 1.0000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1413 | 0.2926 | 2.6624 |  |
| 1.0000 | MPC | yes | 9 | 9 | 0 | 0.1450 | 0.4788 | 2.8659 |  |
| 1.1000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4417 | 0.0928 | 5.7110 |  |
| 1.1000 | MPC | yes | 9 | 9 | 0 | 0.1500 | 0.5267 | 2.8659 |  |
| 1.1000 | 32_32_32 (rich11) | no | 8 | 9 | 1 | 0.1534 | 0.4784 | 2.6624 | unstable__rhp_zero |
| 1.2000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4425 | 0.0982 | 5.7110 |  |
| 1.2000 | MPC | yes | 9 | 9 | 0 | 0.1555 | 0.5746 | 2.8659 |  |
| 1.3000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4433 | 0.1035 | 5.7110 |  |
| 1.3000 | MPC | yes | 9 | 9 | 0 | 0.1615 | 0.6224 | 2.8659 |  |
| 1.4000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4442 | 0.1089 | 5.7110 |  |
| 1.4000 | MPC | yes | 9 | 9 | 0 | 0.1679 | 0.6703 | 2.8659 |  |
| 1.5000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4452 | 0.1142 | 5.7110 |  |
| 1.5000 | MPC | yes | 9 | 9 | 0 | 0.1748 | 0.7182 | 2.8659 |  |
| 1.6000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4463 | 0.1196 | 5.7110 |  |
| 1.6000 | MPC | yes | 9 | 9 | 0 | 0.1823 | 0.7661 | 2.8659 |  |
| 1.7000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4474 | 0.1249 | 5.7110 |  |
| 1.7000 | MPC | yes | 9 | 9 | 0 | 0.1901 | 0.8139 | 2.8659 |  |
| 1.8000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4486 | 0.1303 | 5.7110 |  |
| 1.8000 | MPC | yes | 9 | 9 | 0 | 0.1985 | 0.8618 | 2.8659 |  |
| 1.9000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4498 | 0.1356 | 5.7110 |  |
| 1.9000 | MPC | yes | 9 | 9 | 0 | 0.2073 | 0.9097 | 2.8659 |  |
| 2.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4511 | 0.1410 | 5.7110 |  |
| 2.0000 | MPC | yes | 9 | 9 | 0 | 0.2167 | 0.9576 | 2.8659 |  |
| 3.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4679 | 0.1944 | 5.7110 |  |
| 3.0000 | MPC | yes | 9 | 9 | 0 | 0.3361 | 1.4364 | 2.8659 |  |
| 4.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4911 | 0.2479 | 5.7110 |  |
| 4.0000 | MPC | yes | 9 | 9 | 0 | 0.5034 | 1.9152 | 2.8659 |  |
| 5.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.5208 | 0.3013 | 5.7110 |  |
| 5.0000 | MPC | yes | 9 | 9 | 0 | 0.7185 | 2.3940 | 2.8659 |  |
| 6.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.5570 | 0.3548 | 5.7110 |  |
| 6.0000 | MPC | yes | 9 | 9 | 0 | 0.9813 | 2.8728 | 2.8728 |  |
| 7.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.5998 | 0.4083 | 5.7110 |  |
| 7.0000 | MPC | yes | 9 | 9 | 0 | 1.2920 | 3.3516 | 3.3516 |  |
| 8.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.6490 | 0.4617 | 5.7110 |  |
| 8.0000 | MPC | yes | 9 | 9 | 0 | 1.6504 | 3.8304 | 3.8304 |  |
| 9.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.7047 | 0.5152 | 5.7110 |  |
| 9.0000 | MPC | yes | 9 | 9 | 0 | 2.0566 | 4.3091 | 4.3091 |  |
| 10.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.7669 | 0.5686 | 5.7110 |  |
| 10.0000 | MPC | yes | 9 | 9 | 0 | 2.5106 | 4.7879 | 4.7879 |  |
| 11.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.8357 | 0.6221 | 5.7110 |  |
| 11.0000 | MPC | yes | 9 | 9 | 0 | 3.0124 | 5.2667 | 5.2667 |  |
| 12.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.9109 | 0.6756 | 5.7110 |  |
| 12.0000 | MPC | yes | 9 | 9 | 0 | 3.5620 | 5.7455 | 5.7455 |  |
| 13.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.9926 | 0.7290 | 5.7110 |  |
| 13.0000 | MPC | yes | 9 | 9 | 0 | 4.1594 | 6.2243 | 6.2243 |  |
| 14.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.0809 | 0.7825 | 6.1213 |  |
| 14.0000 | MPC | yes | 9 | 9 | 0 | 4.8046 | 6.7031 | 6.7031 |  |
| 15.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.1756 | 0.8359 | 6.5644 |  |
| 15.0000 | MPC | yes | 9 | 9 | 0 | 5.4976 | 7.1819 | 7.1819 |  |
| 16.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.2769 | 0.8894 | 7.0075 |  |
| 16.0000 | MPC | yes | 9 | 9 | 0 | 6.2384 | 7.6607 | 7.6607 |  |
| 17.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.3846 | 0.9429 | 7.4505 |  |
| 17.0000 | MPC | yes | 9 | 9 | 0 | 7.0269 | 8.1395 | 8.1395 |  |
| 18.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.4989 | 0.9963 | 7.8936 |  |
| 18.0000 | MPC | yes | 9 | 9 | 0 | 7.8633 | 8.6183 | 8.6183 |  |
| 19.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.6196 | 1.0498 | 8.3367 |  |
| 19.0000 | MPC | yes | 9 | 9 | 0 | 8.7474 | 9.0971 | 9.0971 |  |
| 20.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.7468 | 1.1032 | 8.7797 |  |
| 20.0000 | MPC | yes | 9 | 9 | 0 | 9.6793 | 9.5759 | 9.5759 |  |

## Failure breakdown
| scenario | controller | failure_count | failed_plants | failure_reasons | failure_details | late_escalation_plants |
| --- | --- | --- | --- | --- | --- | --- |
| nominal_screen | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=7.531e+05>10; tail_rms=2.477e+05>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.444e+06>10; tail_rms=1.461e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=3.202e+06>10; tail_rms=1.053e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.105e+06>10; tail_rms=1.35e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| step_disturbance_scale_0.4x | 16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.3423>0.2 and tail_g=1.428>1.2) | unstable__rhp_zero |
| step_disturbance_scale_0.6x | 32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.5151>0.2 and tail_g=1.846>1.2) | unstable__rhp_zero |
| step_disturbance_scale_0.9x | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.6263>0.2 and tail_g=2.282>1.2) | unstable__rhp_zero |
| step_disturbance_scale_1.1x | 32_32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.4784>0.2 and tail_g=2.094>1.2) | unstable__rhp_zero |