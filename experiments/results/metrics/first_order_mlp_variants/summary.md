# First-order MLP variant robustness metrics

Study script: `experiments/first_order/compare_first_order_mlp_variants.py`
Nominal comparison horizon: `20 s`
Reference step: `-2`

Scenarios:
- `nominal`: nominal plant, no disturbance
- `disturbed`: measurement noise sigma=0.02, input disturbance sigma=0.02, step 0.1 at 5s
- `hidden_pole`: actual plant has extra hidden pole `-( 10 )`, controllers remain nominal

Metrics (lower is better):
- `rank`: average within-plant/scenario rank across cost, peak, tail RMS, tail growth, max |u|, and failure flag; lower is better
- `cost`: mean rollout stage cost under the same qy/ru/qu objective
- `peak`: maximum absolute tracking error
- `tail_rms`: RMS tracking error over the last 25 percent of the rollout
- `tail_g`: RMS(last 10 percent) / RMS(previous 10 percent)
- `max_u`: maximum absolute control magnitude
- `fails`: number of runs flagged as failed by the threshold rule

## Overall average across all plants and scenarios
| controller | mean_composite_rank | mean_mean_cost | mean_peak_error | mean_tail_rms_error | mean_tail_growth_ratio | mean_max_abs_u | failure_count | late_escalation_count | num_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32_32_32 (rich11) | 4.2438 | 0.1563 | 2.2097 | 0.0651 | 0.8759 | 1.6630 | 1 | 1 | 27 |
| 16_16_16 (rich11) | 4.3179 | 2.232e+11 | 7.334e+05 | 2.412e+05 | 1.6739 | 1.6556 | 3 | 3 | 27 |
| 32_32 (rich11) | 4.4599 | 0.1580 | 2.2082 | 0.0369 | 0.8577 | 1.6796 | 1 | 1 | 27 |
| 16_16 (rich11) | 4.4784 | 2.482e+09 | 6.124e+04 | 2.014e+04 | 1.4510 | 1.6454 | 2 | 2 | 27 |
| 16_16_16 (compact6) | 4.5895 | 2.367e+11 | 9.104e+05 | 2.994e+05 | 1.9660 | 1.6003 | 4 | 4 | 27 |
| 32_32_32 (compact6) | 4.6204 | 4.116e+11 | 1.049e+06 | 3.451e+05 | 1.6326 | 1.6500 | 3 | 3 | 27 |
| 32_32 (compact6) | 4.6265 | 8.424e+07 | 1.066e+04 | 3506.5068 | 1.2000 | 1.7050 | 2 | 2 | 27 |
| 16_16 (compact6) | 4.6636 | 2.366e+11 | 6.300e+05 | 2.072e+05 | 1.5980 | 1.6419 | 3 | 3 | 27 |

Compact text view:
```text
controller           rank    cost       peak       tail_rms   tail_g  max_u   fails
-------------------  ------  ---------  ---------  ---------  ------  ------  -----
32_32_32 (rich11)    4.2438  0.1563     2.2097     0.0651     0.8759  1.6630  1    
16_16_16 (rich11)    4.3179  2.232e+11  7.334e+05  2.412e+05  1.6739  1.6556  3    
32_32 (rich11)       4.4599  0.1580     2.2082     0.0369     0.8577  1.6796  1    
16_16 (rich11)       4.4784  2.482e+09  6.124e+04  2.014e+04  1.4510  1.6454  2    
16_16_16 (compact6)  4.5895  2.367e+11  9.104e+05  2.994e+05  1.9660  1.6003  4    
32_32_32 (compact6)  4.6204  4.116e+11  1.049e+06  3.451e+05  1.6326  1.6500  3    
32_32 (compact6)     4.6265  8.424e+07  1.066e+04  3506.5068  1.2000  1.7050  2    
16_16 (compact6)     4.6636  2.366e+11  6.300e+05  2.072e+05  1.5980  1.6419  3    
```

## Failure breakdown
| scenario | controller | failure_count | failed_plants | failure_reasons | failure_details | late_escalation_plants |
| --- | --- | --- | --- | --- | --- | --- |
| disturbed | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=1.057e+06>10; tail_rms=3.477e+05>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| disturbed | 16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=9.372e+04>10; tail_rms=3.082e+04>0.2 and tail_g=8.165>1.2) | unstable__rhp_zero |
| disturbed | 16_16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.539e+06>10; tail_rms=1.493e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| disturbed | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=2.318e+06>10; tail_rms=7.622e+05>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| disturbed | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=2.879e+05>10; tail_rms=9.467e+04>0.2 and tail_g=8.165>1.2) | unstable__rhp_zero |
| disturbed | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=5.235e+06>10; tail_rms=1.722e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| disturbed | 32_32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=1.369>0.2 and tail_g=1.405>1.2) | unstable__rhp_zero |
| hidden_pole | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=1.52e+07>10; tail_rms=4.999e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| hidden_pole | 16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=1.56e+06>10; tail_rms=5.13e+05>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| hidden_pole | 16_16_16 (compact6) | 2 | unstable__no_zero,unstable__rhp_zero | late_escalation,peak_error | unstable__no_zero (peak_error=1.846e+06>10; tail_rms=6.071e+05>0.2 and tail_g=8.166>1.2); unstable__rhp_zero (peak_error=1.375e+07>10; tail_rms=4.523e+06>0.2 and tail_g=8.166>1.2) | unstable__no_zero,unstable__rhp_zero |
| hidden_pole | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=1.428e+07>10; tail_rms=4.697e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| hidden_pole | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.5623>0.2 and tail_g=2.053>1.2) | unstable__rhp_zero |
| hidden_pole | 32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.3109>0.2 and tail_g=1.354>1.2) | unstable__rhp_zero |
| hidden_pole | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=1.899e+07>10; tail_rms=6.246e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=7.531e+05>10; tail_rms=2.477e+05>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal | 16_16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.444e+06>10; tail_rms=1.461e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=3.202e+06>10; tail_rms=1.053e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |
| nominal | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=4.105e+06>10; tail_rms=1.35e+06>0.2 and tail_g=8.166>1.2) | unstable__rhp_zero |

## Scenario summary
| scenario | controller | mean_composite_rank | mean_mean_cost | mean_peak_error | mean_tail_rms_error | mean_tail_growth_ratio | mean_max_abs_u | failure_count | late_escalation_count | num_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disturbed | 32_32 (rich11) | 4.2593 | 0.1711 | 2.2205 | 0.0470 | 1.2856 | 1.6619 | 0 | 0 | 9 |
| disturbed | 16_16 (rich11) | 4.3704 | 2.679e+07 | 1.042e+04 | 3424.9352 | 2.3732 | 1.6336 | 1 | 1 | 9 |
| disturbed | 16_16_16 (rich11) | 4.3704 | 1.638e+10 | 2.575e+05 | 8.469e+04 | 2.2891 | 1.6267 | 1 | 1 | 9 |
| disturbed | 16_16_16 (compact6) | 4.3889 | 6.283e+10 | 5.043e+05 | 1.659e+05 | 2.3715 | 1.5720 | 1 | 1 | 9 |
| disturbed | 32_32_32 (rich11) | 4.4259 | 0.1968 | 2.2178 | 0.1896 | 1.5140 | 1.6701 | 1 | 1 | 9 |
| disturbed | 32_32 (compact6) | 4.6296 | 2.527e+08 | 3.199e+04 | 1.052e+04 | 2.1969 | 1.7504 | 1 | 1 | 9 |
| disturbed | 16_16 (compact6) | 4.6852 | 3.408e+09 | 1.175e+05 | 3.863e+04 | 2.0167 | 1.6527 | 1 | 1 | 9 |
| disturbed | 32_32_32 (compact6) | 4.8704 | 8.356e+10 | 5.816e+05 | 1.913e+05 | 2.1286 | 1.6456 | 1 | 1 | 9 |
| hidden_pole | 32_32_32 (compact6) | 4.2222 | 1.100e+12 | 2.110e+06 | 6.940e+05 | 1.3576 | 1.7046 | 1 | 1 | 9 |
| hidden_pole | 32_32_32 (rich11) | 4.2407 | 0.1477 | 2.1942 | 0.0023 | 0.5766 | 1.7382 | 0 | 0 | 9 |
| hidden_pole | 16_16_16 (rich11) | 4.3519 | 6.220e+11 | 1.587e+06 | 5.219e+05 | 1.3643 | 1.7294 | 1 | 1 | 9 |
| hidden_pole | 16_16 (rich11) | 4.6111 | 7.418e+09 | 1.733e+05 | 5.700e+04 | 1.4144 | 1.7305 | 1 | 1 | 9 |
| hidden_pole | 16_16 (compact6) | 4.6111 | 7.045e+11 | 1.689e+06 | 5.554e+05 | 1.4014 | 1.6833 | 1 | 1 | 9 |
| hidden_pole | 32_32 (rich11) | 4.6296 | 0.1700 | 2.1910 | 0.0367 | 0.6575 | 1.7701 | 1 | 1 | 9 |
| hidden_pole | 16_16_16 (compact6) | 4.6481 | 5.871e+11 | 1.733e+06 | 5.700e+05 | 2.1664 | 1.6661 | 2 | 2 | 9 |
| hidden_pole | 32_32 (compact6) | 4.6852 | 0.2646 | 2.1951 | 0.0679 | 0.7697 | 1.7604 | 1 | 1 | 9 |
| nominal | 32_32_32 (rich11) | 4.0648 | 0.1245 | 2.2171 | 0.0033 | 0.5371 | 1.5807 | 0 | 0 | 9 |
| nominal | 16_16_16 (rich11) | 4.2315 | 3.126e+10 | 3.557e+05 | 1.170e+05 | 1.3684 | 1.6107 | 1 | 1 | 9 |
| nominal | 16_16 (rich11) | 4.4537 | 0.1306 | 2.1903 | 0.0087 | 0.5654 | 1.5723 | 0 | 0 | 9 |
| nominal | 32_32 (rich11) | 4.4907 | 0.1329 | 2.2130 | 0.0270 | 0.6300 | 1.6067 | 0 | 0 | 9 |
| nominal | 32_32 (compact6) | 4.5648 | 0.1353 | 2.2106 | 0.0163 | 0.6334 | 1.6042 | 0 | 0 | 9 |
| nominal | 16_16 (compact6) | 4.6944 | 1.730e+09 | 8.368e+04 | 2.752e+04 | 1.3760 | 1.5899 | 1 | 1 | 9 |
| nominal | 16_16_16 (compact6) | 4.7315 | 6.021e+10 | 4.937e+05 | 1.624e+05 | 1.3601 | 1.5627 | 1 | 1 | 9 |
| nominal | 32_32_32 (compact6) | 4.7685 | 5.139e+10 | 4.561e+05 | 1.500e+05 | 1.4115 | 1.5999 | 1 | 1 | 9 |