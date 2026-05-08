# First-order robustness metrics

Scenarios:
- `nominal`: nominal plant, no disturbance
- `disturbed`: measurement noise sigma=0.02, input disturbance sigma=0.02, step 0.1 at 5s
- `hidden_pole`: actual plant has extra hidden pole `-( 10 )`, controllers remain nominal

Metrics (lower is better):
- `mean_cost`: rollout mean of the same quadratic stage cost used in the controller objective
- `peak_error`: max absolute tracking error
- `tail_rms_error`: RMS tracking error over the last 25% of the rollout
- `tail_growth_ratio`: RMS(last 10%) / RMS(previous 10%)
- `max_abs_u`: max absolute control action
- `composite_rank`: average rank across the five metrics plus failure flag within each plant/scenario

Compact table columns:
- `rank`: average within-plant/scenario rank across cost, peak, tail RMS, tail growth, max |u|, and failure flag; lower is better
- `cost`: mean rollout stage cost using the same qy/ru/qu objective as training and MPC
- `peak`: maximum absolute tracking error over the rollout
- `tail_rms`: RMS tracking error over the last 25% of the rollout; catches bad end behavior
- `tail_g`: RMS(last 10%) / RMS(previous 10%); above 1 means the error is growing again near the end
- `fails`: number of runs flagged as failed by the threshold rule
- `p95_ms`: 95th percentile controller step time in milliseconds, measured from controller.step(...) only
- `util_%`: runtime utilization = 100 * p95_step_time / dt

Runtime measurement notes:
- controller timing uses replayed `controller.step(...)` calls only
- warmup repeats = 3, measured repeats = 12

Failure rule:
- failure if non-finite, `peak_error > 10`, `max_abs_u > 50`, or late escalation (`tail_rms_error > 0.2` and `tail_growth_ratio > 1.2`)

## Overall average across all plants and scenarios
| controller | mean_composite_rank | mean_mean_cost | mean_peak_error | mean_tail_rms_error | mean_tail_growth_ratio | mean_p95_step_ms | mean_utilization_pct | mean_max_abs_u | failure_count | late_escalation_count | num_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16_16 (compact6) | 5.8673 | 974.1479 | 30.3875 | 13.1618 | 1.0417 | 0.0065 | 0.0065 | 1.6373 | 3 | 3 | 27 |
| 16_16 (rich11) | 5.6759 | 11.3475 | 4.8427 | 1.3785 | 0.9339 | 0.0070 | 0.0070 | 1.6440 | 3 | 3 | 27 |
| 16_16_16 (compact6) | 5.6204 | 1007.1343 | 44.3938 | 19.6123 | 0.9928 | 0.0081 | 0.0081 | 1.5999 | 4 | 4 | 27 |
| 16_16_16 (rich11) | 5.4475 | 963.6448 | 37.8555 | 16.5868 | 0.9601 | 0.0082 | 0.0082 | 1.6579 | 3 | 3 | 27 |
| 32_32 (compact6) | 5.6636 | 0.3554 | 2.2079 | 0.1386 | 0.9828 | 0.0074 | 0.0074 | 1.6801 | 3 | 3 | 27 |
| 32_32 (rich11) | 5.3549 | 0.2936 | 2.2057 | 0.0794 | 0.8857 | 0.0075 | 0.0075 | 1.6797 | 0 | 0 | 27 |
| 32_32_32 (compact6) | 5.9290 | 1709.7611 | 50.2430 | 22.2547 | 1.0577 | 0.0092 | 0.0092 | 1.6535 | 3 | 3 | 27 |
| 32_32_32 (rich11) | 5.2006 | 0.2683 | 2.2107 | 0.0412 | 0.7857 | 0.0092 | 0.0092 | 1.6476 | 0 | 0 | 27 |
| MPC | 4.4784 | 0.2603 | 2.2622 | 0.0417 | 0.4565 | 0.0211 | 0.0211 | 1.8054 | 0 | 0 | 27 |
| RL (PID-features) | 5.7623 | 1.0579 | 2.7095 | 0.0430 | 0.5316 | 0.0024 | 0.0024 | 2.4849 | 0 | 0 | 27 |

Compact text view:
```text
controller           rank    cost       peak     tail_rms  tail_g  p95_ms  util_%  fails
-------------------  ------  ---------  -------  --------  ------  ------  ------  -----
16_16 (compact6)     5.8673  974.1479   30.3875  13.1618   1.0417  0.0065  0.0065  3    
16_16 (rich11)       5.6759  11.3475    4.8427   1.3785    0.9339  0.0070  0.0070  3    
16_16_16 (compact6)  5.6204  1007.1343  44.3938  19.6123   0.9928  0.0081  0.0081  4    
16_16_16 (rich11)    5.4475  963.6448   37.8555  16.5868   0.9601  0.0082  0.0082  3    
32_32 (compact6)     5.6636  0.3554     2.2079   0.1386    0.9828  0.0074  0.0074  3    
32_32 (rich11)       5.3549  0.2936     2.2057   0.0794    0.8857  0.0075  0.0075  0    
32_32_32 (compact6)  5.9290  1709.7611  50.2430  22.2547   1.0577  0.0092  0.0092  3    
32_32_32 (rich11)    5.2006  0.2683     2.2107   0.0412    0.7857  0.0092  0.0092  0    
MPC                  4.4784  0.2603     2.2622   0.0417    0.4565  0.0211  0.0211  0    
RL (PID-features)    5.7623  1.0579     2.7095   0.0430    0.5316  0.0024  0.0024  0    
```

## Failure breakdown
| scenario | controller | failure_count | failed_plants | failure_reasons | failure_details | late_escalation_plants |
| --- | --- | --- | --- | --- | --- | --- |
| disturbed | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=43.54>10; tail_rms=20.26>0.2 and tail_g=2.829>1.2) | unstable__rhp_zero |
| disturbed | 16_16 (rich11) | 2 | integrator__rhp_zero,unstable__rhp_zero | late_escalation | integrator__rhp_zero (tail_rms=0.2934>0.2 and tail_g=1.211>1.2); unstable__rhp_zero (tail_rms=2.503>0.2 and tail_g=1.898>1.2) | integrator__rhp_zero,unstable__rhp_zero |
| disturbed | 16_16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=238.1>10; tail_rms=109.7>0.2 and tail_g=2.974>1.2) | unstable__rhp_zero |
| disturbed | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=174.3>10; tail_rms=80.46>0.2 and tail_g=2.951>1.2) | unstable__rhp_zero |
| disturbed | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=1.842>0.2 and tail_g=1.492>1.2) | unstable__rhp_zero |
| disturbed | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=255.9>10; tail_rms=117.6>0.2 and tail_g=2.994>1.2) | unstable__rhp_zero |
| hidden_pole | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=690.8>10; tail_rms=317.6>0.2 and tail_g=2.995>1.2) | unstable__rhp_zero |
| hidden_pole | 16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=72.23>10; tail_rms=33.76>0.2 and tail_g=2.828>1.2) | unstable__rhp_zero |
| hidden_pole | 16_16_16 (compact6) | 2 | unstable__no_zero,unstable__rhp_zero | late_escalation,peak_error | unstable__no_zero (peak_error=83.14>10; tail_rms=38>0.2 and tail_g=3.052>1.2); unstable__rhp_zero (peak_error=625.4>10; tail_rms=287.6>0.2 and tail_g=2.99>1.2) | unstable__no_zero,unstable__rhp_zero |
| hidden_pole | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=649.7>10; tail_rms=298.9>0.2 and tail_g=2.987>1.2) | unstable__rhp_zero |
| hidden_pole | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.9582>0.2 and tail_g=4.157>1.2) | unstable__rhp_zero |
| hidden_pole | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=862.5>10; tail_rms=396.3>0.2 and tail_g=3.001>1.2) | unstable__rhp_zero |
| nominal | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=34.93>10; tail_rms=16.34>0.2 and tail_g=2.823>1.2) | unstable__rhp_zero |
| nominal | 16_16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=202.8>10; tail_rms=93.53>0.2 and tail_g=2.96>1.2) | unstable__rhp_zero |
| nominal | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=146.7>10; tail_rms=67.84>0.2 and tail_g=2.93>1.2) | unstable__rhp_zero |
| nominal | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.2218>0.2 and tail_g=1.804>1.2) | unstable__rhp_zero |
| nominal | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=186.7>10; tail_rms=85.9>0.2 and tail_g=2.982>1.2) | unstable__rhp_zero |

## Scenario averages across plants
| scenario | controller | mean_composite_rank | mean_mean_cost | mean_peak_error | mean_tail_rms_error | mean_tail_growth_ratio | mean_p95_step_ms | mean_utilization_pct | mean_max_abs_u | failure_count | late_escalation_count | num_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disturbed | 16_16 (compact6) | 5.7407 | 12.2080 | 6.7338 | 2.3271 | 1.0362 | 0.0067 | 0.0067 | 1.6387 | 1 | 1 | 9 |
| disturbed | 16_16 (rich11) | 5.5926 | 0.4599 | 2.4022 | 0.3430 | 0.9767 | 0.0075 | 0.0075 | 1.6294 | 2 | 2 | 9 |
| disturbed | 16_16_16 (compact6) | 5.1481 | 346.6122 | 28.3576 | 12.2240 | 1.0998 | 0.0080 | 0.0080 | 1.5709 | 1 | 1 | 9 |
| disturbed | 16_16_16 (rich11) | 5.3889 | 186.7701 | 21.2729 | 9.0026 | 1.0743 | 0.0083 | 0.0083 | 1.6335 | 1 | 1 | 9 |
| disturbed | 32_32 (compact6) | 5.4630 | 0.4015 | 2.2180 | 0.2653 | 0.9874 | 0.0073 | 0.0073 | 1.6757 | 1 | 1 | 9 |
| disturbed | 32_32 (rich11) | 4.9630 | 0.2953 | 2.2131 | 0.1417 | 0.9521 | 0.0074 | 0.0074 | 1.6622 | 0 | 0 | 9 |
| disturbed | 32_32_32 (compact6) | 6.1296 | 398.4844 | 30.3415 | 13.1579 | 1.1100 | 0.0092 | 0.0092 | 1.6561 | 1 | 1 | 9 |
| disturbed | 32_32_32 (rich11) | 5.1667 | 0.2634 | 2.2206 | 0.0973 | 0.9296 | 0.0091 | 0.0091 | 1.6237 | 0 | 0 | 9 |
| disturbed | MPC | 5.7037 | 0.2568 | 2.2772 | 0.1249 | 1.0215 | 0.0211 | 0.0211 | 1.7552 | 0 | 0 | 9 |
| disturbed | RL (PID-features) | 5.7037 | 0.8792 | 2.5931 | 0.0667 | 0.7594 | 0.0024 | 0.0024 | 2.3617 | 0 | 0 | 9 |
| hidden_pole | 16_16 (compact6) | 5.8426 | 2902.1667 | 78.6514 | 35.3212 | 1.0076 | 0.0066 | 0.0066 | 1.6833 | 1 | 1 | 9 |
| hidden_pole | 16_16 (rich11) | 5.7500 | 33.3232 | 9.9355 | 3.7711 | 0.9744 | 0.0066 | 0.0066 | 1.7305 | 1 | 1 | 9 |
| hidden_pole | 16_16_16 (compact6) | 5.8796 | 2422.6120 | 80.3956 | 36.2017 | 1.0808 | 0.0081 | 0.0081 | 1.6661 | 2 | 2 | 9 |
| hidden_pole | 16_16_16 (rich11) | 5.5463 | 2571.2483 | 74.0950 | 33.2169 | 0.9085 | 0.0081 | 0.0081 | 1.7294 | 1 | 1 | 9 |
| hidden_pole | 32_32 (compact6) | 5.7870 | 0.3977 | 2.1951 | 0.1162 | 1.1149 | 0.0075 | 0.0075 | 1.7604 | 1 | 1 | 9 |
| hidden_pole | 32_32 (rich11) | 5.6389 | 0.3264 | 2.1910 | 0.0740 | 0.9010 | 0.0075 | 0.0075 | 1.7701 | 0 | 0 | 9 |
| hidden_pole | 32_32_32 (compact6) | 5.4907 | 4518.1299 | 97.7405 | 44.0454 | 1.0260 | 0.0092 | 0.0092 | 1.7046 | 1 | 1 | 9 |
| hidden_pole | 32_32_32 (rich11) | 5.3796 | 0.2938 | 2.1942 | 0.0150 | 0.7302 | 0.0093 | 0.0093 | 1.7382 | 0 | 0 | 9 |
| hidden_pole | MPC | 4.0648 | 0.2830 | 2.2353 | 1.672e-04 | 0.2029 | 0.0211 | 0.0211 | 1.9420 | 0 | 0 | 9 |
| hidden_pole | RL (PID-features) | 5.6204 | 1.4248 | 2.9452 | 0.0378 | 0.3988 | 0.0024 | 0.0024 | 2.7442 | 0 | 0 | 9 |
| nominal | 16_16 (compact6) | 6.0185 | 8.0691 | 5.7773 | 1.8372 | 1.0811 | 0.0063 | 0.0063 | 1.5899 | 1 | 1 | 9 |
| nominal | 16_16 (rich11) | 5.6852 | 0.2594 | 2.1903 | 0.0215 | 0.8504 | 0.0068 | 0.0068 | 1.5723 | 0 | 0 | 9 |
| nominal | 16_16_16 (compact6) | 5.8333 | 252.1788 | 24.4281 | 10.4113 | 0.7976 | 0.0080 | 0.0080 | 1.5627 | 1 | 1 | 9 |
| nominal | 16_16_16 (rich11) | 5.4074 | 132.9159 | 18.1986 | 7.5408 | 0.8975 | 0.0082 | 0.0082 | 1.6107 | 1 | 1 | 9 |
| nominal | 32_32 (compact6) | 5.7407 | 0.2669 | 2.2106 | 0.0344 | 0.8459 | 0.0074 | 0.0074 | 1.6042 | 1 | 1 | 9 |
| nominal | 32_32 (rich11) | 5.4630 | 0.2592 | 2.2130 | 0.0226 | 0.8041 | 0.0075 | 0.0075 | 1.6067 | 0 | 0 | 9 |
| nominal | 32_32_32 (compact6) | 6.1667 | 212.6690 | 22.6469 | 9.5609 | 1.0372 | 0.0091 | 0.0091 | 1.5999 | 1 | 1 | 9 |
| nominal | 32_32_32 (rich11) | 5.0556 | 0.2478 | 2.2171 | 0.0114 | 0.6972 | 0.0091 | 0.0091 | 1.5807 | 0 | 0 | 9 |
| nominal | MPC | 3.6667 | 0.2410 | 2.2740 | 1.396e-07 | 0.1450 | 0.0212 | 0.0212 | 1.7189 | 0 | 0 | 9 |
| nominal | RL (PID-features) | 5.9630 | 0.8697 | 2.5900 | 0.0246 | 0.4367 | 0.0022 | 0.0022 | 2.3489 | 0 | 0 | 9 |

Compact text view:
```text
scenario     controller           rank    cost       peak     tail_rms   tail_g  p95_ms  util_%  fails
-----------  -------------------  ------  ---------  -------  ---------  ------  ------  ------  -----
disturbed    16_16 (compact6)     5.7407  12.2080    6.7338   2.3271     1.0362  0.0067  0.0067  1    
disturbed    16_16 (rich11)       5.5926  0.4599     2.4022   0.3430     0.9767  0.0075  0.0075  2    
disturbed    16_16_16 (compact6)  5.1481  346.6122   28.3576  12.2240    1.0998  0.0080  0.0080  1    
disturbed    16_16_16 (rich11)    5.3889  186.7701   21.2729  9.0026     1.0743  0.0083  0.0083  1    
disturbed    32_32 (compact6)     5.4630  0.4015     2.2180   0.2653     0.9874  0.0073  0.0073  1    
disturbed    32_32 (rich11)       4.9630  0.2953     2.2131   0.1417     0.9521  0.0074  0.0074  0    
disturbed    32_32_32 (compact6)  6.1296  398.4844   30.3415  13.1579    1.1100  0.0092  0.0092  1    
disturbed    32_32_32 (rich11)    5.1667  0.2634     2.2206   0.0973     0.9296  0.0091  0.0091  0    
disturbed    MPC                  5.7037  0.2568     2.2772   0.1249     1.0215  0.0211  0.0211  0    
disturbed    RL (PID-features)    5.7037  0.8792     2.5931   0.0667     0.7594  0.0024  0.0024  0    
hidden_pole  16_16 (compact6)     5.8426  2902.1667  78.6514  35.3212    1.0076  0.0066  0.0066  1    
hidden_pole  16_16 (rich11)       5.7500  33.3232    9.9355   3.7711     0.9744  0.0066  0.0066  1    
hidden_pole  16_16_16 (compact6)  5.8796  2422.6120  80.3956  36.2017    1.0808  0.0081  0.0081  2    
hidden_pole  16_16_16 (rich11)    5.5463  2571.2483  74.0950  33.2169    0.9085  0.0081  0.0081  1    
hidden_pole  32_32 (compact6)     5.7870  0.3977     2.1951   0.1162     1.1149  0.0075  0.0075  1    
hidden_pole  32_32 (rich11)       5.6389  0.3264     2.1910   0.0740     0.9010  0.0075  0.0075  0    
hidden_pole  32_32_32 (compact6)  5.4907  4518.1299  97.7405  44.0454    1.0260  0.0092  0.0092  1    
hidden_pole  32_32_32 (rich11)    5.3796  0.2938     2.1942   0.0150     0.7302  0.0093  0.0093  0    
hidden_pole  MPC                  4.0648  0.2830     2.2353   1.672e-04  0.2029  0.0211  0.0211  0    
hidden_pole  RL (PID-features)    5.6204  1.4248     2.9452   0.0378     0.3988  0.0024  0.0024  0    
nominal      16_16 (compact6)     6.0185  8.0691     5.7773   1.8372     1.0811  0.0063  0.0063  1    
nominal      16_16 (rich11)       5.6852  0.2594     2.1903   0.0215     0.8504  0.0068  0.0068  0    
nominal      16_16_16 (compact6)  5.8333  252.1788   24.4281  10.4113    0.7976  0.0080  0.0080  1    
nominal      16_16_16 (rich11)    5.4074  132.9159   18.1986  7.5408     0.8975  0.0082  0.0082  1    
nominal      32_32 (compact6)     5.7407  0.2669     2.2106   0.0344     0.8459  0.0074  0.0074  1    
nominal      32_32 (rich11)       5.4630  0.2592     2.2130   0.0226     0.8041  0.0075  0.0075  0    
nominal      32_32_32 (compact6)  6.1667  212.6690   22.6469  9.5609     1.0372  0.0091  0.0091  1    
nominal      32_32_32 (rich11)    5.0556  0.2478     2.2171   0.0114     0.6972  0.0091  0.0091  0    
nominal      MPC                  3.6667  0.2410     2.2740   1.396e-07  0.1450  0.0212  0.0212  0    
nominal      RL (PID-features)    5.9630  0.8697     2.5900   0.0246     0.4367  0.0022  0.0022  0    
```
