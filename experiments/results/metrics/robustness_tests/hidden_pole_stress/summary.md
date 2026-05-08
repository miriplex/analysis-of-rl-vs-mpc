# Hidden-pole stress test

The stress plant is defined as:

`G_real(s) = G_nominal(s) * (10 / (s + 10))^n`

Method:
- Screen all controllers on the nominal first-order plant set.
- Only controllers with zero nominal failures are eligible for the hidden-pole stress sweep.
- Increase hidden-pole order `n = 1, 2, ...` until no eligible controller passes all plants.

Rollout horizon: `10 s`
Reference step: `-2`
Hidden pole location: `100`
Tested hidden orders: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]`

## Nominal screening
| controller | pass_all_plants | pass_count | num_plants | failure_count | mean_cost | mean_tail_rms_error | failed_plants |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MPC | yes | 9 | 9 | 0 | 0.2410 | 1.396e-07 |  |
| 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2478 | 0.0114 |  |
| 16_16 (rich11) | yes | 9 | 9 | 0 | 0.2594 | 0.0215 |  |
| 32_32 (rich11) | yes | 9 | 9 | 0 | 0.2592 | 0.0226 |  |
| RL (PID-features) | yes | 9 | 9 | 0 | 0.8697 | 0.0246 |  |
| 32_32 (compact6) | no | 8 | 9 | 1 | 0.2669 | 0.0344 | unstable__rhp_zero |
| 16_16 (compact6) | no | 8 | 9 | 1 | 8.0691 | 1.8372 | unstable__rhp_zero |
| 16_16_16 (rich11) | no | 8 | 9 | 1 | 132.9159 | 7.5408 | unstable__rhp_zero |
| 32_32_32 (compact6) | no | 8 | 9 | 1 | 212.6690 | 9.5609 | unstable__rhp_zero |
| 16_16_16 (compact6) | no | 8 | 9 | 1 | 252.1788 | 10.4113 | unstable__rhp_zero |

## Survival summary
| controller | eligible_after_nominal | max_pass_all_order | first_failure_order | failed_plants_at_first_failure | failure_reasons_at_first_failure | worst_tail_rms_at_first_failure |
| --- | --- | --- | --- | --- | --- | --- |
| 32_32 (rich11) | yes | 17 | 18 | unstable__rhp_zero | late_escalation | 2.9455 |
| 32_32_32 (rich11) | yes | 12 | 13 | unstable__rhp_zero | late_escalation | 0.9053 |
| RL (PID-features) | yes | 10 | 11 | unstable__rhp_zero | peak_error | 0.3659 |
| MPC | yes | 10 | 11 | unstable__rhp_zero | late_escalation,max_abs_u,peak_error | 2.468e+04 |
| 16_16 (rich11) | yes | 8 | 9 | unstable__rhp_zero | late_escalation | 1.7354 |
| 16_16 (compact6) | no | 0 |  |  |  | 0 |
| 16_16_16 (compact6) | no | 0 |  |  |  | 0 |
| 16_16_16 (rich11) | no | 0 |  |  |  | 0 |
| 32_32 (compact6) | no | 0 |  |  |  | 0 |
| 32_32_32 (compact6) | no | 0 |  |  |  | 0 |

## Order-by-order progression
| hidden_order | controller | pass_all_plants | pass_count | num_plants | failure_count | mean_cost | worst_tail_rms_error | worst_peak_error | failed_plants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MPC | yes | 9 | 9 | 0 | 0.2448 | 4.032e-07 | 2.8917 |  |
| 1 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2512 | 0.0527 | 2.6863 |  |
| 1 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.2631 | 0.0587 | 2.5919 |  |
| 1 | RL (PID-features) | yes | 9 | 9 | 0 | 0.8950 | 0.1057 | 5.8527 |  |
| 1 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.2629 | 0.1339 | 2.6675 |  |
| 2 | MPC | yes | 9 | 9 | 0 | 0.2487 | 5.035e-07 | 2.9169 |  |
| 2 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2547 | 0.0516 | 2.7115 |  |
| 2 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.2670 | 0.0627 | 2.6047 |  |
| 2 | RL (PID-features) | yes | 9 | 9 | 0 | 0.9223 | 0.1055 | 5.9939 |  |
| 2 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.2668 | 0.1347 | 2.6956 |  |
| 3 | MPC | yes | 9 | 9 | 0 | 0.2529 | 1.110e-06 | 2.9395 |  |
| 3 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2584 | 0.0506 | 2.7352 |  |
| 3 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.2711 | 0.0672 | 2.6298 |  |
| 3 | RL (PID-features) | yes | 9 | 9 | 0 | 0.9517 | 0.1053 | 6.1269 |  |
| 3 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.2708 | 0.1372 | 2.7241 |  |
| 4 | MPC | yes | 9 | 9 | 0 | 0.2571 | 2.592e-06 | 2.9541 |  |
| 4 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2623 | 0.0494 | 2.7541 |  |
| 4 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.2753 | 0.0726 | 2.6658 |  |
| 4 | RL (PID-features) | yes | 9 | 9 | 0 | 0.9830 | 0.1051 | 6.3868 |  |
| 4 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.2750 | 0.1411 | 2.7512 |  |
| 5 | MPC | yes | 9 | 9 | 0 | 0.2615 | 5.220e-06 | 2.9685 |  |
| 5 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2663 | 0.0479 | 2.7671 |  |
| 5 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.2799 | 0.0801 | 2.6982 |  |
| 5 | RL (PID-features) | yes | 9 | 9 | 0 | 1.0156 | 0.1049 | 6.7278 |  |
| 5 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.2796 | 0.1468 | 2.7751 |  |
| 6 | MPC | yes | 9 | 9 | 0 | 0.2660 | 9.417e-06 | 2.9849 |  |
| 6 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2705 | 0.0457 | 2.7906 |  |
| 6 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.2852 | 0.0926 | 2.7239 |  |
| 6 | RL (PID-features) | yes | 9 | 9 | 0 | 1.0509 | 0.1047 | 7.1542 |  |
| 6 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.2847 | 0.1598 | 2.7933 |  |
| 7 | MPC | yes | 9 | 9 | 0 | 0.2706 | 1.581e-05 | 2.9885 |  |
| 7 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2751 | 0.0423 | 2.8123 |  |
| 7 | RL (PID-features) | yes | 9 | 9 | 0 | 1.0948 | 0.1045 | 7.6144 |  |
| 7 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.2922 | 0.1155 | 2.7314 |  |
| 7 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.2906 | 0.1827 | 2.8035 |  |
| 8 | MPC | yes | 9 | 9 | 0 | 0.2758 | 2.542e-05 | 2.9757 |  |
| 8 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2806 | 0.0368 | 2.8287 |  |
| 8 | RL (PID-features) | yes | 9 | 9 | 0 | 1.1594 | 0.1043 | 8.1752 |  |
| 8 | 16_16 (rich11) | yes | 9 | 9 | 0 | 0.3044 | 0.1746 | 2.7341 |  |
| 8 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.2980 | 0.2246 | 2.8046 |  |
| 9 | MPC | yes | 9 | 9 | 0 | 0.2821 | 4.133e-05 | 2.9501 |  |
| 9 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2872 | 0.0288 | 2.8334 |  |
| 9 | RL (PID-features) | yes | 9 | 9 | 0 | 1.2623 | 0.1041 | 8.8200 |  |
| 9 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.3080 | 0.3059 | 2.7964 |  |
| 9 | 16_16 (rich11) | no | 8 | 9 | 1 | 0.4212 | 1.7354 | 2.7599 | unstable__rhp_zero |
| 10 | MPC | yes | 9 | 9 | 0 | 0.2948 | 6.857e-04 | 2.9179 |  |
| 10 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.2961 | 0.0329 | 2.8212 |  |
| 10 | RL (PID-features) | yes | 9 | 9 | 0 | 1.4294 | 0.1039 | 9.5041 |  |
| 10 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.3232 | 0.4627 | 2.7800 |  |
| 11 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.3095 | 0.0725 | 2.7963 |  |
| 11 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.3467 | 0.7059 | 2.7568 |  |
| 11 | RL (PID-features) | no | 8 | 9 | 1 | 1.7073 | 0.3659 | 10.1969 | unstable__rhp_zero |
| 11 | MPC | no | 8 | 9 | 1 | 9.161e+10 | 2.468e+04 | 6.518e+04 | unstable__rhp_zero |
| 12 | 32_32_32 (rich11) | yes | 9 | 9 | 0 | 0.3339 | 0.1736 | 2.7684 |  |
| 12 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.3828 | 1.0156 | 2.7307 |  |
| 13 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.4315 | 1.3522 | 2.7457 |  |
| 13 | 32_32_32 (rich11) | no | 8 | 9 | 1 | 0.4681 | 0.9053 | 2.7440 | unstable__rhp_zero |
| 14 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.4773 | 1.6277 | 2.7615 |  |
| 15 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.5160 | 1.8442 | 2.7759 |  |
| 16 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.5848 | 2.2301 | 3.0358 |  |
| 17 | 32_32 (rich11) | yes | 9 | 9 | 0 | 0.6910 | 2.7653 | 3.4803 |  |
| 18 | 32_32 (rich11) | no | 8 | 9 | 1 | 0.8054 | 2.9455 | 4.3252 | unstable__rhp_zero |

## Failure breakdown
| scenario | controller | failure_count | failed_plants | failure_reasons | failure_details | late_escalation_plants |
| --- | --- | --- | --- | --- | --- | --- |
| hidden_order_11 | MPC | 1 | unstable__rhp_zero | late_escalation,max_abs_u,peak_error | unstable__rhp_zero (peak_error=6.518e+04>10; max_abs_u=1.801e+06>50; tail_rms=2.468e+04>0.2 and tail_g=5.572>1.2) | unstable__rhp_zero |
| hidden_order_11 | RL (PID-features) | 1 | unstable__rhp_zero | peak_error | unstable__rhp_zero (peak_error=10.2>10) |  |
| hidden_order_13 | 32_32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.9053>0.2 and tail_g=2.563>1.2) | unstable__rhp_zero |
| hidden_order_18 | 32_32 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=2.945>0.2 and tail_g=1.963>1.2) | unstable__rhp_zero |
| hidden_order_9 | 16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=1.735>0.2 and tail_g=1.441>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=34.93>10; tail_rms=16.34>0.2 and tail_g=2.823>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=202.8>10; tail_rms=93.53>0.2 and tail_g=2.96>1.2) | unstable__rhp_zero |
| nominal_screen | 16_16_16 (rich11) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=146.7>10; tail_rms=67.84>0.2 and tail_g=2.93>1.2) | unstable__rhp_zero |
| nominal_screen | 32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation | unstable__rhp_zero (tail_rms=0.2218>0.2 and tail_g=1.804>1.2) | unstable__rhp_zero |
| nominal_screen | 32_32_32 (compact6) | 1 | unstable__rhp_zero | late_escalation,peak_error | unstable__rhp_zero (peak_error=186.7>10; tail_rms=85.9>0.2 and tail_g=2.982>1.2) | unstable__rhp_zero |