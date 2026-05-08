# Measurement-noise stress test

Method:
- Screen all controllers on the nominal first-order plant set.
- Only controllers with zero nominal failures are eligible for the measurement-noise stress sweep.
- Increase the measurement-noise standard deviation by a multiplier `x` while keeping input disturbance noise and step disturbance at zero.
- Base measurement noise standard deviation: `0.02`

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
| RL (PID-features) | yes | 3.0000 | 4.0000 | unstable__rhp_zero | late_escalation | 0.2082 |
| 16_16 (rich11) | yes | 2.0000 | 3.0000 | unstable__rhp_zero | late_escalation,peak_error | 3950.1464 |
| 32_32_32 (rich11) | yes | 2.0000 | 3.0000 | unstable__rhp_zero | late_escalation,peak_error | 2.755e+04 |
| 32_32 (compact6) | yes | 0.7000 | 0.8000 | unstable__rhp_zero | late_escalation | 0.7000 |
| 32_32 (rich11) | yes | 0 | 0.1000 | unstable__rhp_zero | late_escalation | 0.3230 |
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
| 0.1000 | MPC | yes | 9 | 9 | 0 | 0.1211 | 0.0040 | 2.8655 |  |
| 0.1000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1246 | 0.0164 | 2.6635 |  |
| 0.1000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1305 | 0.0297 | 2.5873 |  |
| 0.1000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4370 | 0.0341 | 5.7091 |  |
| 0.1000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1393 | 0.1566 | 2.6637 |  |
| 0.1000 | 32_32 (rich11) | no | 8 | 9 | 1 | 0.1390 | 0.3230 | 2.6544 | unstable__rhp_zero |
| 0.2000 | MPC | yes | 9 | 9 | 0 | 0.1211 | 0.0081 | 2.8652 |  |
| 0.2000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1246 | 0.0188 | 2.6645 |  |
| 0.2000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1305 | 0.0300 | 2.5895 |  |
| 0.2000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4368 | 0.0343 | 5.7072 |  |
| 0.2000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1448 | 0.2250 | 2.6633 |  |
| 0.3000 | MPC | yes | 9 | 9 | 0 | 0.1212 | 0.0121 | 2.8648 |  |
| 0.3000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1247 | 0.0219 | 2.6656 |  |
| 0.3000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1305 | 0.0304 | 2.5917 |  |
| 0.3000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4366 | 0.0345 | 5.7054 |  |
| 0.3000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1474 | 0.2484 | 2.6630 |  |
| 0.4000 | MPC | yes | 9 | 9 | 0 | 0.1212 | 0.0162 | 2.8645 |  |
| 0.4000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1248 | 0.0255 | 2.6666 |  |
| 0.4000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1306 | 0.0309 | 2.5939 |  |
| 0.4000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4365 | 0.0347 | 5.7035 |  |
| 0.4000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1478 | 0.2496 | 2.6626 |  |
| 0.5000 | MPC | yes | 9 | 9 | 0 | 0.1213 | 0.0202 | 2.8641 |  |
| 0.5000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1249 | 0.0295 | 2.6677 |  |
| 0.5000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1306 | 0.0314 | 2.5960 |  |
| 0.5000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4363 | 0.0350 | 5.7016 |  |
| 0.5000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1505 | 0.2745 | 2.6623 |  |
| 0.6000 | MPC | yes | 9 | 9 | 0 | 0.1214 | 0.0243 | 2.8638 |  |
| 0.6000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1307 | 0.0320 | 2.5982 |  |
| 0.6000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1250 | 0.0337 | 2.6688 |  |
| 0.6000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4362 | 0.0353 | 5.6997 |  |
| 0.6000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1488 | 0.2570 | 2.6619 |  |
| 0.7000 | MPC | yes | 9 | 9 | 0 | 0.1215 | 0.0283 | 2.8634 |  |
| 0.7000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1308 | 0.0327 | 2.6004 |  |
| 0.7000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4362 | 0.0364 | 5.6979 |  |
| 0.7000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1252 | 0.0381 | 2.6698 |  |
| 0.7000 | 32_32 (compact6) | yes | 9 | 9 | 0 | 0.1503 | 0.2697 | 2.6616 |  |
| 0.8000 | MPC | yes | 9 | 9 | 0 | 0.1216 | 0.0324 | 2.8631 |  |
| 0.8000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1310 | 0.0335 | 2.6025 |  |
| 0.8000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4361 | 0.0416 | 5.6960 |  |
| 0.8000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1253 | 0.0428 | 2.6709 |  |
| 0.8000 | 32_32 (compact6) | no | 8 | 9 | 1 | 0.2325 | 0.7000 | 2.6612 | unstable__rhp_zero |
| 0.9000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1312 | 0.0343 | 2.6047 |  |
| 0.9000 | MPC | yes | 9 | 9 | 0 | 0.1218 | 0.0364 | 2.8627 |  |
| 0.9000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4361 | 0.0469 | 5.6941 |  |
| 0.9000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1256 | 0.0476 | 2.6719 |  |
| 1.0000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1315 | 0.0365 | 2.6069 |  |
| 1.0000 | MPC | yes | 9 | 9 | 0 | 0.1220 | 0.0405 | 2.8624 |  |
| 1.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4361 | 0.0521 | 5.6922 |  |
| 1.0000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1258 | 0.0527 | 2.6730 |  |
| 1.1000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1318 | 0.0422 | 2.6091 |  |
| 1.1000 | MPC | yes | 9 | 9 | 0 | 0.1221 | 0.0445 | 2.8620 |  |
| 1.1000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4361 | 0.0573 | 5.6904 |  |
| 1.1000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1261 | 0.0579 | 2.6740 |  |
| 1.2000 | MPC | yes | 9 | 9 | 0 | 0.1223 | 0.0486 | 2.8617 |  |
| 1.2000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1321 | 0.0488 | 2.6112 |  |
| 1.2000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4362 | 0.0625 | 5.6885 |  |
| 1.2000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1264 | 0.0633 | 2.6751 |  |
| 1.3000 | MPC | yes | 9 | 9 | 0 | 0.1226 | 0.0526 | 2.8614 |  |
| 1.3000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1325 | 0.0564 | 2.6134 |  |
| 1.3000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4363 | 0.0677 | 5.6866 |  |
| 1.3000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1268 | 0.0689 | 2.6761 |  |
| 1.4000 | MPC | yes | 9 | 9 | 0 | 0.1228 | 0.0567 | 2.8610 |  |
| 1.4000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1330 | 0.0646 | 2.6155 |  |
| 1.4000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4364 | 0.0729 | 5.6848 |  |
| 1.4000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1272 | 0.0746 | 2.6772 |  |
| 1.5000 | MPC | yes | 9 | 9 | 0 | 0.1231 | 0.0607 | 2.8607 |  |
| 1.5000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1336 | 0.0731 | 2.6177 |  |
| 1.5000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4365 | 0.0781 | 5.6829 |  |
| 1.5000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1276 | 0.0805 | 2.6782 |  |
| 1.6000 | MPC | yes | 9 | 9 | 0 | 0.1233 | 0.0648 | 2.8603 |  |
| 1.6000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1343 | 0.0816 | 2.6198 |  |
| 1.6000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4367 | 0.0833 | 5.6810 |  |
| 1.6000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1281 | 0.0865 | 2.6793 |  |
| 1.7000 | MPC | yes | 9 | 9 | 0 | 0.1236 | 0.0688 | 2.8600 |  |
| 1.7000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4369 | 0.0885 | 5.6791 |  |
| 1.7000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1350 | 0.0894 | 2.6220 |  |
| 1.7000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1287 | 0.0927 | 2.6803 |  |
| 1.8000 | MPC | yes | 9 | 9 | 0 | 0.1239 | 0.0729 | 2.8596 |  |
| 1.8000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4371 | 0.0937 | 5.6773 |  |
| 1.8000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1359 | 0.0963 | 2.6241 |  |
| 1.8000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1294 | 0.0989 | 2.6813 |  |
| 1.9000 | MPC | yes | 9 | 9 | 0 | 0.1243 | 0.0769 | 2.8593 |  |
| 1.9000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4374 | 0.0989 | 5.6754 |  |
| 1.9000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1369 | 0.1041 | 2.6263 |  |
| 1.9000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1301 | 0.1050 | 2.6824 |  |
| 2.0000 | MPC | yes | 9 | 9 | 0 | 0.1246 | 0.0810 | 2.8589 |  |
| 2.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4376 | 0.1041 | 5.6735 |  |
| 2.0000 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.1309 | 0.1110 | 2.6834 |  |
| 2.0000 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.1383 | 0.1276 | 2.6284 |  |
| 3.0000 | MPC | yes | 9 | 9 | 0 | 0.1290 | 0.1214 | 2.8736 |  |
| 3.0000 | RL (PID-features) | yes | 9 | 9 | 0 | 0.4418 | 0.1562 | 5.6701 |  |
| 3.0000 | 16_16 (rich11) | no | 8 | 9 | 1 | 4.399e+05 | 3950.1464 | 1.201e+04 | unstable__rhp_zero |
| 3.0000 | 32_32_32 (rich11) | no | 8 | 9 | 1 | 2.140e+07 | 2.755e+04 | 8.377e+04 | unstable__rhp_zero |
| 4.0000 | MPC | yes | 9 | 9 | 0 | 0.1352 | 0.1619 | 2.9024 |  |
| 4.0000 | RL (PID-features) | no | 8 | 9 | 1 | 0.4485 | 0.2082 | 5.6797 | unstable__rhp_zero |
| 5.0000 | MPC | yes | 9 | 9 | 0 | 0.1431 | 0.2024 | 2.9312 |  |
| 6.0000 | MPC | yes | 9 | 9 | 0 | 0.1528 | 0.2429 | 2.9601 |  |
| 7.0000 | MPC | yes | 9 | 9 | 0 | 0.1642 | 0.2833 | 2.9889 |  |
| 8.0000 | MPC | yes | 9 | 9 | 0 | 0.1774 | 0.3238 | 3.0177 |  |
| 9.0000 | MPC | yes | 9 | 9 | 0 | 0.1924 | 0.3643 | 3.0465 |  |
| 10.0000 | MPC | yes | 9 | 9 | 0 | 0.2091 | 0.4048 | 3.0754 |  |
| 11.0000 | MPC | yes | 9 | 9 | 0 | 0.2276 | 0.4452 | 3.1042 |  |
| 12.0000 | MPC | yes | 9 | 9 | 0 | 0.2479 | 0.4857 | 3.1330 |  |
| 13.0000 | MPC | yes | 9 | 9 | 0 | 0.2699 | 0.5262 | 3.1618 |  |
| 14.0000 | MPC | yes | 9 | 9 | 0 | 0.2937 | 0.5667 | 3.1906 |  |
| 15.0000 | MPC | yes | 9 | 9 | 0 | 0.3192 | 0.6071 | 3.2195 |  |
| 16.0000 | MPC | yes | 9 | 9 | 0 | 0.3465 | 0.6476 | 3.2483 |  |
| 17.0000 | MPC | yes | 9 | 9 | 0 | 0.3756 | 0.6881 | 3.2771 |  |
| 18.0000 | MPC | yes | 9 | 9 | 0 | 0.4064 | 0.7286 | 3.3059 |  |
| 19.0000 | MPC | yes | 9 | 9 | 0 | 0.4390 | 0.7690 | 3.3348 |  |
| 20.0000 | MPC | yes | 9 | 9 | 0 | 0.4734 | 0.8095 | 3.3636 |  |

## Failure breakdown
| scenario | controller | failure_count | failed_plants | failure_reasons | failure_details | late_escalation_plants |
| --- | --- | --- | --- | --- | --- | --- |
| measurement_noise_scale_0.1x | 32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.323>0.2 and tail_g=2.188>1.2) | unstable__rhp_zero |
| measurement_noise_scale_0.8x | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.7>0.2 and tail_g=1.855>1.2) | unstable__rhp_zero |
| measurement_noise_scale_3x | 16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=1.201e+04>10; tail_rms=3950>0.2 and tail_g=8.155>1.2) | unstable__rhp_zero |
| measurement_noise_scale_3x | 32_32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=8.377e+04>10; tail_rms=2.755e+04>0.2 and tail_g=8.163>1.2) | unstable__rhp_zero |
| measurement_noise_scale_4x | RL (PID-features) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.2082>0.2 and tail_g=1.328>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=7.531e+05>10; tail_rms=2.477e+05>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.444e+06>10; tail_rms=1.461e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=3.202e+06>10; tail_rms=1.053e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal_screen | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.105e+06>10; tail_rms=1.35e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |