# Input-noise stress test

Method:
- Screen all controllers on the nominal first-order plant set.
- Only controllers with zero nominal failures are eligible for the input-noise stress sweep.
- Increase the input disturbance-noise standard deviation by a multiplier `x` while keeping measurement noise and step disturbance at zero.
- Base input disturbance-noise standard deviation: `0.02`

Rollout horizon: `20 s`
Reference step: `-2`
Tested stress scales: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0]`

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
| RL (PID-features) | yes | 44.0000 | 45.0000 | unstable__no_zero | late_escalation | 2.9823 |
| MPC | yes | 42.0000 | 43.0000 | unstable__no_zero | late_escalation | 2.5438 |
| 32_32_32 (rich11) | yes | 2.0000 | 3.0000 | unstable__rhp_zero | late_escalation,peak_error | 9.250e+05 |
| 16_16 (rich11) | yes | 1.9000 | 2.0000 | unstable__rhp_zero | late_escalation,peak_error | 9.7277 |
| 32_32 (compact6) | yes | 0.4000 | 0.5000 | unstable__rhp_zero | late_escalation,peak_error | 558.6590 |
| 32_32 (rich11) | yes | 0 | 0.1000 | unstable__rhp_zero | late_escalation | 0.3165 |
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
| 0.1000 | MPC | yes | 9 | 9 | 0 | 0.1211 | 0.0059 | 2.8668 |  |
| 0.1000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1246 | 0.0175 | 2.6647 |  |
| 0.1000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1308 | 0.0300 | 2.5823 |  |
| 0.1000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4372 | 0.0343 | 5.7077 |  |
| 0.1000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1399 | 0.1596 | 2.6649 |  |
| 0.1000 | 32_32 (rich11) | no | 8 | 9 | 1 | 0.1389 | 0.3165 | 2.6532 | unstable__rhp_zero |
| 0.2000 | MPC | yes | 9 | 9 | 0 | 0.1212 | 0.0118 | 2.8677 |  |
| 0.2000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1247 | 0.0214 | 2.6670 |  |
| 0.2000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1310 | 0.0308 | 2.5794 |  |
| 0.2000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4371 | 0.0347 | 5.7045 |  |
| 0.2000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1476 | 0.2435 | 2.6658 |  |
| 0.3000 | MPC | yes | 9 | 9 | 0 | 0.1213 | 0.0177 | 2.8686 |  |
| 0.3000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1249 | 0.0261 | 2.6693 |  |
| 0.3000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1313 | 0.0317 | 2.5766 |  |
| 0.3000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4372 | 0.0352 | 5.7013 |  |
| 0.3000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1678 | 0.3773 | 2.6667 |  |
| 0.4000 | MPC | yes | 9 | 9 | 0 | 0.1215 | 0.0237 | 2.8695 |  |
| 0.4000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1251 | 0.0314 | 2.6715 |  |
| 0.4000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1317 | 0.0327 | 2.5737 |  |
| 0.4000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4373 | 0.0360 | 5.6980 |  |
| 0.4000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1536 | 0.2797 | 2.6676 |  |
| 0.5000 | MPC | yes | 9 | 9 | 0 | 0.1217 | 0.0296 | 2.8704 |  |
| 0.5000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1322 | 0.0340 | 2.5726 |  |
| 0.5000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4375 | 0.0369 | 5.6948 |  |
| 0.5000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1254 | 0.0369 | 2.6738 |  |
| 0.5000 | 32_32 (compact6) | no | 8 | 9 | 1 | 8799.6441 | 558.6590 | 1695.8545 | unstable__rhp_zero |
| 0.6000 | MPC | yes | 9 | 9 | 0 | 0.1220 | 0.0355 | 2.8713 |  |
| 0.6000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1328 | 0.0367 | 2.5726 |  |
| 0.6000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4377 | 0.0398 | 5.6915 |  |
| 0.6000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1257 | 0.0426 | 2.6761 |  |
| 0.7000 | MPC | yes | 9 | 9 | 0 | 0.1224 | 0.0414 | 2.8722 |  |
| 0.7000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1334 | 0.0418 | 2.5726 |  |
| 0.7000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4380 | 0.0464 | 5.6883 |  |
| 0.7000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1261 | 0.0486 | 2.6784 |  |
| 0.8000 | MPC | yes | 9 | 9 | 0 | 0.1228 | 0.0473 | 2.8731 |  |
| 0.8000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1342 | 0.0475 | 2.5726 |  |
| 0.8000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4383 | 0.0530 | 5.6850 |  |
| 0.8000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1265 | 0.0547 | 2.6807 |  |
| 0.9000 | MPC | yes | 9 | 9 | 0 | 0.1232 | 0.0532 | 2.8740 |  |
| 0.9000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1351 | 0.0541 | 2.5726 |  |
| 0.9000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4387 | 0.0596 | 5.6818 |  |
| 0.9000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1270 | 0.0609 | 2.6830 |  |
| 1.0000 | MPC | yes | 9 | 9 | 0 | 0.1237 | 0.0592 | 2.8749 |  |
| 1.0000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1361 | 0.0621 | 2.5726 |  |
| 1.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4392 | 0.0663 | 5.6785 |  |
| 1.0000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1276 | 0.0673 | 2.6853 |  |
| 1.1000 | MPC | yes | 9 | 9 | 0 | 0.1243 | 0.0651 | 2.8758 |  |
| 1.1000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1373 | 0.0723 | 2.5726 |  |
| 1.1000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4397 | 0.0729 | 5.6753 |  |
| 1.1000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1282 | 0.0738 | 2.6876 |  |
| 1.2000 | MPC | yes | 9 | 9 | 0 | 0.1249 | 0.0710 | 2.8767 |  |
| 1.2000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4403 | 0.0795 | 5.6720 |  |
| 1.2000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1289 | 0.0804 | 2.6898 |  |
| 1.2000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1387 | 0.0858 | 2.5726 |  |
| 1.3000 | MPC | yes | 9 | 9 | 0 | 0.1256 | 0.0769 | 2.8776 |  |
| 1.3000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4409 | 0.0862 | 5.6688 |  |
| 1.3000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1297 | 0.0871 | 2.6921 |  |
| 1.3000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1404 | 0.1042 | 2.5725 |  |
| 1.4000 | MPC | yes | 9 | 9 | 0 | 0.1263 | 0.0828 | 2.8785 |  |
| 1.4000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4416 | 0.0928 | 5.6655 |  |
| 1.4000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1306 | 0.0938 | 2.6944 |  |
| 1.4000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1424 | 0.1281 | 2.5725 |  |
| 1.5000 | MPC | yes | 9 | 9 | 0 | 0.1270 | 0.0887 | 2.8794 |  |
| 1.5000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4424 | 0.0994 | 5.6623 |  |
| 1.5000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1315 | 0.1005 | 2.6967 |  |
| 1.5000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1449 | 0.1540 | 2.5725 |  |
| 1.6000 | MPC | yes | 9 | 9 | 0 | 0.1279 | 0.0947 | 2.8803 |  |
| 1.6000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4432 | 0.1060 | 5.6590 |  |
| 1.6000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1326 | 0.1072 | 2.6990 |  |
| 1.6000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1478 | 0.1806 | 2.5725 |  |
| 1.7000 | MPC | yes | 9 | 9 | 0 | 0.1287 | 0.1006 | 2.8813 |  |
| 1.7000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4440 | 0.1127 | 5.6558 |  |
| 1.7000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1337 | 0.1137 | 2.7013 |  |
| 1.7000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1515 | 0.2349 | 2.5725 |  |
| 1.8000 | MPC | yes | 9 | 9 | 0 | 0.1297 | 0.1065 | 2.8822 |  |
| 1.8000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4449 | 0.1193 | 5.6525 |  |
| 1.8000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1351 | 0.1196 | 2.7036 |  |
| 1.8000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1572 | 0.3657 | 2.5725 |  |
| 1.9000 | MPC | yes | 9 | 9 | 0 | 0.1306 | 0.1124 | 2.8831 |  |
| 1.9000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1368 | 0.1244 | 2.7059 |  |
| 1.9000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4459 | 0.1259 | 5.6495 |  |
| 1.9000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1671 | 0.6087 | 2.5725 |  |
| 2.0000 | MPC | yes | 9 | 9 | 0 | 0.1317 | 0.1183 | 2.8840 |  |
| 2.0000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1391 | 0.1273 | 2.7082 |  |
| 2.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4469 | 0.1325 | 5.6499 |  |
| 2.0000 | 16_16 (rich11) | no | 8 | 9 | 1 | 2.8401 | 9.7277 | 28.1471 | unstable__rhp_zero |
| 3.0000 | MPC | yes | 9 | 9 | 0 | 0.1449 | 0.1775 | 2.8930 |  |
| 3.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4604 | 0.1988 | 5.6542 |  |
| 3.0000 | 32_32_32 (rich11) | no | 8 | 9 | 1 | 2.412e+10 | 9.250e+05 | 2.813e+06 | unstable__rhp_zero |
| 4.0000 | MPC | yes | 9 | 9 | 0 | 0.1634 | 0.2366 | 2.9021 |  |
| 4.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4797 | 0.2651 | 5.6585 |  |
| 5.0000 | MPC | yes | 9 | 9 | 0 | 0.1872 | 0.2958 | 2.9111 |  |
| 5.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.5047 | 0.3314 | 5.6628 |  |
| 6.0000 | MPC | yes | 9 | 9 | 0 | 0.2163 | 0.3550 | 2.9244 |  |
| 6.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.5355 | 0.3976 | 5.6671 |  |
| 7.0000 | MPC | yes | 9 | 9 | 0 | 0.2507 | 0.4141 | 2.9473 |  |
| 7.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.5720 | 0.4639 | 5.6714 |  |
| 8.0000 | MPC | yes | 9 | 9 | 0 | 0.2904 | 0.4733 | 2.9702 |  |
| 8.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.6143 | 0.5302 | 5.6757 |  |
| 9.0000 | MPC | yes | 9 | 9 | 0 | 0.3354 | 0.5324 | 2.9931 |  |
| 9.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.6623 | 0.5965 | 5.6800 |  |
| 10.0000 | MPC | yes | 9 | 9 | 0 | 0.3856 | 0.5916 | 3.0160 |  |
| 10.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.7161 | 0.6627 | 5.6843 |  |
| 11.0000 | MPC | yes | 9 | 9 | 0 | 0.4412 | 0.6507 | 3.0389 |  |
| 11.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.7756 | 0.7290 | 5.6886 |  |
| 12.0000 | MPC | yes | 9 | 9 | 0 | 0.5020 | 0.7099 | 3.0617 |  |
| 12.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.8409 | 0.7953 | 5.6929 |  |
| 13.0000 | MPC | yes | 9 | 9 | 0 | 0.5681 | 0.7691 | 3.0846 |  |
| 13.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.9120 | 0.8615 | 5.6971 |  |
| 14.0000 | MPC | yes | 9 | 9 | 0 | 0.6396 | 0.8282 | 3.1075 |  |
| 14.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.9888 | 0.9278 | 5.7014 |  |
| 15.0000 | MPC | yes | 9 | 9 | 0 | 0.7163 | 0.8874 | 3.1304 |  |
| 15.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.0713 | 0.9941 | 5.7057 |  |
| 16.0000 | MPC | yes | 9 | 9 | 0 | 0.7983 | 0.9465 | 3.1533 |  |
| 16.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.1597 | 1.0604 | 5.7100 |  |
| 17.0000 | MPC | yes | 9 | 9 | 0 | 0.8856 | 1.0057 | 3.1762 |  |
| 17.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.2537 | 1.1266 | 5.7143 |  |
| 18.0000 | MPC | yes | 9 | 9 | 0 | 0.9782 | 1.0649 | 3.1991 |  |
| 18.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.3536 | 1.1929 | 5.7696 |  |
| 19.0000 | MPC | yes | 9 | 9 | 0 | 1.0761 | 1.1240 | 3.2219 |  |
| 19.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.4591 | 1.2592 | 5.8605 |  |
| 20.0000 | MPC | yes | 9 | 9 | 0 | 1.1792 | 1.1832 | 3.2448 |  |
| 20.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.5705 | 1.3255 | 5.9515 |  |
| 21.0000 | MPC | yes | 9 | 9 | 0 | 1.2877 | 1.2423 | 3.2677 |  |
| 21.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.6876 | 1.3917 | 6.0424 |  |
| 22.0000 | MPC | yes | 9 | 9 | 0 | 1.4014 | 1.3015 | 3.2906 |  |
| 22.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.8104 | 1.4580 | 6.1334 |  |
| 23.0000 | MPC | yes | 9 | 9 | 0 | 1.5205 | 1.3606 | 3.3135 |  |
| 23.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 1.9390 | 1.5243 | 6.2243 |  |
| 24.0000 | MPC | yes | 9 | 9 | 0 | 1.6448 | 1.4198 | 3.3364 |  |
| 24.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 2.0734 | 1.5905 | 6.3153 |  |
| 25.0000 | MPC | yes | 9 | 9 | 0 | 1.7744 | 1.4790 | 3.3592 |  |
| 25.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 2.2135 | 1.6568 | 6.4063 |  |
| 26.0000 | MPC | yes | 9 | 9 | 0 | 1.9093 | 1.5381 | 3.3821 |  |
| 26.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 2.3593 | 1.7231 | 6.4972 |  |
| 27.0000 | MPC | yes | 9 | 9 | 0 | 2.0495 | 1.5973 | 3.4050 |  |
| 27.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 2.5110 | 1.7894 | 6.5882 |  |
| 28.0000 | MPC | yes | 9 | 9 | 0 | 2.1950 | 1.6564 | 3.5096 |  |
| 28.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 2.6683 | 1.8556 | 6.6791 |  |
| 29.0000 | MPC | yes | 9 | 9 | 0 | 2.3458 | 1.7156 | 3.6350 |  |
| 29.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 2.8315 | 1.9219 | 6.7701 |  |
| 30.0000 | MPC | yes | 9 | 9 | 0 | 2.5019 | 1.7748 | 3.7603 |  |
| 30.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 3.0004 | 1.9882 | 6.8610 |  |
| 31.0000 | MPC | yes | 9 | 9 | 0 | 2.6633 | 1.8339 | 3.8857 |  |
| 31.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 3.1750 | 2.0545 | 6.9520 |  |
| 32.0000 | MPC | yes | 9 | 9 | 0 | 2.8299 | 1.8931 | 4.0110 |  |
| 32.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 3.3554 | 2.1207 | 7.0429 |  |
| 33.0000 | MPC | yes | 9 | 9 | 0 | 3.0019 | 1.9522 | 4.1364 |  |
| 33.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 3.5416 | 2.1870 | 7.1339 |  |
| 34.0000 | MPC | yes | 9 | 9 | 0 | 3.1791 | 2.0114 | 4.2617 |  |
| 34.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 3.7335 | 2.2533 | 7.2248 |  |
| 35.0000 | MPC | yes | 9 | 9 | 0 | 3.3616 | 2.0705 | 4.3871 |  |
| 35.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 3.9311 | 2.3196 | 7.3158 |  |
| 36.0000 | MPC | yes | 9 | 9 | 0 | 3.5495 | 2.1297 | 4.5124 |  |
| 36.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 4.1346 | 2.3858 | 7.4067 |  |
| 37.0000 | MPC | yes | 9 | 9 | 0 | 3.7426 | 2.1889 | 4.6377 |  |
| 37.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 4.3437 | 2.4521 | 7.4977 |  |
| 38.0000 | MPC | yes | 9 | 9 | 0 | 3.9410 | 2.2480 | 4.7631 |  |
| 38.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 4.5587 | 2.5184 | 7.5886 |  |
| 39.0000 | MPC | yes | 9 | 9 | 0 | 4.1447 | 2.3072 | 4.8884 |  |
| 39.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 4.7793 | 2.5846 | 7.6796 |  |
| 40.0000 | MPC | yes | 9 | 9 | 0 | 4.3536 | 2.3663 | 5.0138 |  |
| 40.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 5.0058 | 2.6509 | 7.7706 |  |
| 41.0000 | MPC | yes | 9 | 9 | 0 | 4.5679 | 2.4255 | 5.1391 |  |
| 41.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 5.2380 | 2.7172 | 7.8615 |  |
| 42.0000 | MPC | yes | 9 | 9 | 0 | 4.7875 | 2.4847 | 5.2645 |  |
| 42.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 5.4759 | 2.7835 | 7.9525 |  |
| 43.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 5.7196 | 2.8497 | 8.0434 |  |
| 43.0000 | MPC | no | 8 | 9 | 1 | 5.0123 | 2.5438 | 5.3898 | unstable__no_zero |
| 44.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 5.9691 | 2.9160 | 8.1344 |  |
| 45.0000 | RL (PID-features) | no | 8 | 9 | 1 | 6.2243 | 2.9823 | 8.2253 | unstable__no_zero |

## Failure breakdown
| scenario | controller | failure_count | failed_plants | failure_reasons | failure_details | late_escalation_plants |
| --- | --- | --- | --- | --- | --- | --- |
| input_noise_scale_0.1x | 32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.3165>0.2 and tail_g=1.715>1.2) | unstable__rhp_zero |
| input_noise_scale_0.5x | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=1696>10; tail_rms=558.7>0.2 and tail_g=7.996>1.2) | unstable__rhp_zero |
| input_noise_scale_2x | 16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=28.15>10; tail_rms=9.728>0.2 and tail_g=5.22>1.2) | unstable__rhp_zero |
| input_noise_scale_3x | 32_32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=2.813e+06>10; tail_rms=9.25e+05>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| input_noise_scale_43x | MPC | 1 | unstable__no_zero | late_escalation | unstable__no_zero (tail_rms=0.2041>0.2 and tail_g=1.45>1.2) | unstable__no_zero |
| input_noise_scale_45x | RL (PID-features) | 1 | unstable__no_zero | late_escalation | unstable__no_zero (tail_rms=0.203>0.2 and tail_g=1.507>1.2) | unstable__no_zero |
| nominal_screen | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=7.531e+05>10; tail_rms=2.477e+05>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.444e+06>10; tail_rms=1.461e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=3.202e+06>10; tail_rms=1.053e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.105e+06>10; tail_rms=1.35e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |