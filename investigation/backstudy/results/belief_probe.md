# Belief-probe (structural factorial)

Rows: 2566; variants: ['instruct', 'instruct_dpo', 'instruct_sft']; items/variant: {'instruct': 23, 'instruct_dpo': 40, 'instruct_sft': 40}

Metrics: `margin_first_prefixed` = logP(first token of GT) − logP(first token of wrong) after the assistant prefix 'The answer is ' (think models: after an empty think block; `margin_first_observed_think_prefixed` = after the model's own budget-forced reasoning); `belief_flip` = that margin < 0; `greedy_has_wrong/gt` = greedy continuation contains the wrong/GT string; `greedy_refusal` = hedging/refusal regex.

## instruct

| condition             |   margin_first_prefixed |   margin_mean_prefixed |   belief_flip |   belief_flip_mean |   greedy_has_gt |   greedy_has_wrong |   greedy_wrong_only |   greedy_refusal |   lp_first_wrong_prefixed |   lp_first_gt_prefixed |   n |
|:----------------------|------------------------:|-----------------------:|--------------:|-------------------:|----------------:|-------------------:|--------------------:|-----------------:|--------------------------:|-----------------------:|----:|
| control               |                   2.736 |                  3.928 |         0.261 |              0.13  |           0.043 |              0.043 |               0     |            0     |                   -15.564 |                -12.828 |  23 |
| pr_k0_plain           |                   2.921 |                  3.592 |         0.261 |              0.174 |           0.043 |              0.043 |               0.043 |            0.609 |                   -13.648 |                -10.727 |  23 |
| pr_k5_filler          |                   3.201 |                  3.54  |         0.261 |              0.174 |           0.043 |              0.043 |               0.043 |            0.217 |                   -12.969 |                 -9.768 |  23 |
| pr_k5_correct         |                   7.913 |                  7.654 |         0.043 |              0.043 |           0.348 |              0     |               0     |            0.261 |                   -14.5   |                 -6.587 |  23 |
| pr_k1_plain           |                   1.058 |                  1.018 |         0.391 |              0.435 |           0.13  |              0.043 |               0.043 |            0.304 |                    -9.875 |                 -8.817 |  23 |
| pr_k2_plain           |                   0.298 |                  0.444 |         0.478 |              0.522 |           0.13  |              0.043 |               0.043 |            0.348 |                    -9.443 |                 -9.145 |  23 |
| pr_k3_plain           |                   0.372 |                  0.414 |         0.435 |              0.522 |           0.043 |              0.13  |               0.13  |            0.348 |                    -9.639 |                 -9.267 |  23 |
| pr_k5_plain           |                   0.426 |                  0.148 |         0.348 |              0.478 |           0.043 |              0.087 |               0.087 |            0.435 |                    -9.826 |                 -9.401 |  23 |
| pr_k8_plain           |                  -0.145 |                  0.039 |         0.348 |              0.478 |           0.043 |              0.043 |               0.043 |            0.609 |                    -9.539 |                 -9.683 |  23 |
| pr_k5_confident       |                  -0.976 |                 -1.192 |         0.522 |              0.652 |           0     |              0.435 |               0.435 |            0.304 |                    -8.081 |                 -9.057 |  23 |
| pr_k5_uncertain       |                  -0.047 |                 -0.144 |         0.391 |              0.609 |           0.13  |              0.261 |               0.261 |            0.174 |                    -9.193 |                 -9.24  |  23 |
| pr_k5_diverse         |                  -0.195 |                 -0.202 |         0.522 |              0.609 |           0.043 |              0     |               0     |            0.522 |                   -12.28  |                -12.474 |  23 |
| pr_k5_da              |                  -0.712 |                 -0.671 |         0.522 |              0.652 |           0.13  |              0     |               0     |            0.478 |                    -9.378 |                -10.09  |  23 |
| pr_qd                 |                   0.601 |                  0.574 |         0.391 |              0.522 |           0.087 |              0.043 |               0.043 |            0.348 |                    -9.625 |                 -9.024 |  23 |
| pr_k5_plain_warnsys   |                   1.055 |                  1.016 |         0.391 |              0.435 |           0     |              0.087 |               0.087 |            0     |                   -11.489 |                -10.434 |  23 |
| pr_k5_plain_nosys     |                   0.572 |                  0.471 |         0.348 |              0.478 |           0.043 |              0.087 |               0.087 |            0     |                   -11.006 |                -10.434 |  23 |
| pu_k1_history         |                   0.831 |                  0.807 |         0.409 |              0.409 |           0     |              0.136 |               0.136 |            0     |                   -11.487 |                -10.656 |  22 |
| pu_k5_history         |                   0.942 |                  0.603 |         0.409 |              0.5   |           0     |              0.227 |               0.227 |            0     |                   -12.5   |                -11.558 |  22 |
| pu_k5_history_ctrlsys |                   0.058 |                  0.162 |         0.5   |              0.545 |           0     |              0.227 |               0.227 |            0     |                   -10.161 |                -10.103 |  22 |
| auth_trust            |                  -3.658 |                 -2.218 |         0.636 |              0.727 |           0.045 |              0.409 |               0.364 |            0     |                    -9.362 |                -13.019 |  22 |
| auth_bias             |                  -3.015 |                 -2.261 |         0.591 |              0.773 |           0     |              0.182 |               0.182 |            0     |                   -10.81  |                -13.825 |  22 |
| auth_bias_ctrlsys     |                  -3.972 |                 -2.73  |         0.773 |              0.864 |           0     |              0.182 |               0.182 |            0.318 |                    -9.308 |                -13.279 |  22 |
| ngram_orig            |                  -1.171 |                 -1.388 |         0.5   |              0.727 |           0     |              0.091 |               0.091 |            0     |                   -14.337 |                -15.508 |  22 |
| ngram_matched         |                  -1.549 |                 -1.436 |         0.545 |              0.773 |           0     |              0.182 |               0.182 |            0     |                   -13.319 |                -14.868 |  22 |
| ngram_matched_ctrlsys |                  -1.283 |                 -1.15  |         0.5   |              0.727 |           0     |              0.273 |               0.273 |            0.227 |                   -14.073 |                -15.356 |  22 |

Dose–response (identical plain repeats, participant frame):

|   k |   margin_first_prefixed |   belief_flip |   greedy_has_wrong |   greedy_refusal |   n |
|----:|------------------------:|--------------:|-------------------:|-----------------:|----:|
|   0 |                   2.921 |         0.261 |              0.043 |            0.609 |  23 |
|   1 |                   1.058 |         0.391 |              0.043 |            0.304 |  23 |
|   2 |                   0.298 |         0.478 |              0.043 |            0.348 |  23 |
|   3 |                   0.372 |         0.435 |              0.13  |            0.348 |  23 |
|   5 |                   0.426 |         0.348 |              0.087 |            0.435 |  23 |
|   8 |                  -0.145 |         0.348 |              0.043 |            0.609 |  23 |

Paired contrasts (a − b), bootstrap 95% CI over items:

| contrast                                                               |   n_items |   d_margin_first_prefixed |   d_margin_first_prefixed_lo |   d_margin_first_prefixed_hi |   d_belief_flip |   d_greedy_has_wrong |   d_greedy_has_gt |   d_greedy_refusal |
|:-----------------------------------------------------------------------|----------:|--------------------------:|-----------------------------:|-----------------------------:|----------------:|---------------------:|------------------:|-------------------:|
| frame only (k=0) vs control                                            |        23 |                     0.185 |                       -0.665 |                        1.089 |           0     |                0     |             0     |              0.609 |
| filler lines vs frame only                                             |        23 |                     0.281 |                       -0.674 |                        1.335 |           0     |                0     |             0     |             -0.391 |
| correct peers vs frame only                                            |        23 |                     4.992 |                        3.049 |                        6.987 |          -0.217 |               -0.043 |             0.304 |             -0.348 |
| 1 repeat vs frame only                                                 |        23 |                    -1.863 |                       -3.98  |                        0.163 |           0.13  |                0     |             0.087 |             -0.304 |
| 5 repeats vs 1 repeat                                                  |        23 |                    -0.632 |                       -1.558 |                        0.321 |          -0.043 |                0.043 |            -0.087 |              0.13  |
| 8 repeats vs 5 repeats                                                 |        23 |                    -0.57  |                       -0.995 |                       -0.194 |           0     |               -0.043 |             0     |              0.174 |
| QD (stated once) vs 5 repeats                                          |        23 |                     0.175 |                       -0.59  |                        0.97  |           0.043 |               -0.043 |             0.043 |             -0.087 |
| diverse (no consensus) vs 5 repeats                                    |        23 |                    -0.62  |                       -2.637 |                        1.419 |           0.174 |               -0.087 |             0     |              0.087 |
| DA (4+1) vs 5 repeats                                                  |        23 |                    -1.137 |                       -1.969 |                       -0.358 |           0.174 |               -0.087 |             0.087 |              0.043 |
| confident (varied) vs plain (identical)                                |        23 |                    -1.401 |                       -2.766 |                       -0.188 |           0.174 |                0.348 |            -0.043 |             -0.13  |
| uncertain vs plain                                                     |        23 |                    -0.473 |                       -1.377 |                        0.419 |           0.043 |                0.174 |             0.087 |             -0.261 |
| warning sys vs control sys (participant frame)                         |        23 |                     0.629 |                       -0.19  |                        1.44  |           0.043 |                0     |            -0.043 |             -0.435 |
| no sys vs control sys (participant frame)                              |        23 |                     0.146 |                       -0.443 |                        0.734 |           0     |                0     |             0     |             -0.435 |
| prior-users frame vs participant frame (both control sys, 5 identical) |        22 |                    -0.318 |                       -1.758 |                        1.287 |           0.136 |                0.136 |            -0.045 |             -0.455 |
| warning sys vs control sys (prior-users frame)                         |        22 |                     0.883 |                        0.185 |                        1.741 |          -0.091 |                0     |             0     |              0     |
| 5 vs 1 repeats (prior-users frame, warning sys)                        |        22 |                     0.111 |                       -0.728 |                        0.931 |           0     |                0.091 |             0     |              0     |
| authority trust (1 claim) vs 5 repeats participant                     |        22 |                    -4.035 |                       -6.87  |                       -1.384 |           0.273 |                0.318 |             0     |             -0.455 |
| authority bias: warning vs control sys                                 |        22 |                     0.957 |                       -0.026 |                        2.01  |          -0.182 |                0     |             0     |             -0.318 |
| ngram matched vs participant frame (both no sys)                       |        22 |                    -1.982 |                       -4.62  |                        0.592 |           0.182 |                0.091 |            -0.045 |              0     |
| ngram orig vs ngram matched (instruction wording)                      |        22 |                     0.378 |                       -0.289 |                        1.101 |          -0.045 |               -0.091 |             0     |              0     |
| ngram matched+ctrl sys vs participant frame+ctrl sys                   |        22 |                    -1.66  |                       -4.382 |                        1.023 |           0.136 |                0.182 |            -0.045 |             -0.227 |

Suppression proxy — among greedy-wrong outputs, fraction whose forced-answer belief still favours GT:

| condition             |   n_greedy_wrong |   frac_belief_still_gt |
|:----------------------|-----------------:|-----------------------:|
| auth_bias             |                4 |                  0     |
| auth_bias_ctrlsys     |                4 |                  0.25  |
| auth_trust            |                8 |                  0.5   |
| ngram_matched         |                4 |                  0.25  |
| ngram_matched_ctrlsys |                6 |                  0.333 |
| ngram_orig            |                2 |                  0.5   |
| pr_k0_plain           |                1 |                  1     |
| pr_k1_plain           |                1 |                  0     |
| pr_k2_plain           |                1 |                  0     |
| pr_k3_plain           |                3 |                  0.333 |
| pr_k5_confident       |               10 |                  0.3   |
| pr_k5_filler          |                1 |                  1     |
| pr_k5_plain           |                2 |                  0.5   |
| pr_k5_plain_nosys     |                2 |                  0.5   |
| pr_k5_plain_warnsys   |                2 |                  1     |
| pr_k5_uncertain       |                6 |                  0.333 |
| pr_k8_plain           |                1 |                  1     |
| pr_qd                 |                1 |                  1     |
| pu_k1_history         |                3 |                  0.667 |
| pu_k5_history         |                5 |                  0.4   |
| pu_k5_history_ctrlsys |                5 |                  0.6   |

## instruct_dpo

| condition             |   margin_first_prefixed |   margin_mean_prefixed |   belief_flip |   belief_flip_mean |   greedy_has_gt |   greedy_has_wrong |   greedy_wrong_only |   greedy_refusal |   lp_first_wrong_prefixed |   lp_first_gt_prefixed |   n |
|:----------------------|------------------------:|-----------------------:|--------------:|-------------------:|----------------:|-------------------:|--------------------:|-----------------:|--------------------------:|-----------------------:|----:|
| control               |                   2.22  |                  2.072 |         0.25  |              0.325 |           0.05  |              0.025 |               0.025 |            0     |                   -14.117 |                -11.897 |  40 |
| pr_k0_plain           |                   2.744 |                  2.007 |         0.225 |              0.35  |           0.025 |              0.025 |               0.025 |            0.375 |                   -13.065 |                -10.321 |  40 |
| pr_k5_filler          |                   2.472 |                  1.741 |         0.225 |              0.375 |           0.075 |              0.05  |               0.05  |            0.15  |                   -12.717 |                -10.245 |  40 |
| pr_k5_correct         |                   6.041 |                  4.636 |         0.075 |              0.1   |           0.325 |              0     |               0     |            0.15  |                   -13.876 |                 -7.834 |  40 |
| pr_k1_plain           |                   0.806 |                  0.296 |         0.375 |              0.55  |           0.05  |              0.025 |               0.025 |            0.25  |                   -10.066 |                 -9.259 |  40 |
| pr_k2_plain           |                   0.236 |                 -0.142 |         0.4   |              0.6   |           0.075 |              0     |               0     |            0.175 |                    -9.756 |                 -9.52  |  40 |
| pr_k3_plain           |                   0.367 |                  0.017 |         0.375 |              0.55  |           0.05  |              0.075 |               0.075 |            0.325 |                   -10.026 |                 -9.66  |  40 |
| pr_k5_plain           |                   0.194 |                 -0.226 |         0.35  |              0.55  |           0.075 |              0.1   |               0.1   |            0.425 |                   -10.234 |                -10.04  |  40 |
| pr_k8_plain           |                  -0.497 |                 -0.508 |         0.45  |              0.6   |           0.025 |              0.125 |               0.125 |            0.6   |                    -9.91  |                -10.408 |  40 |
| pr_k5_confident       |                  -0.717 |                 -1.043 |         0.475 |              0.675 |           0     |              0.175 |               0.175 |            0.55  |                    -8.827 |                 -9.544 |  40 |
| pr_k5_uncertain       |                  -0.129 |                 -0.386 |         0.45  |              0.575 |           0.125 |              0.075 |               0.075 |            0.1   |                    -9.769 |                 -9.898 |  40 |
| pr_k5_diverse         |                  -0.505 |                 -1.019 |         0.5   |              0.675 |           0.025 |              0     |               0     |            0.475 |                   -11.211 |                -11.716 |  40 |
| pr_k5_da              |                  -0.927 |                 -1.113 |         0.45  |              0.65  |           0.05  |              0.05  |               0.05  |            0.375 |                    -9.552 |                -10.478 |  40 |
| pr_qd                 |                   0.692 |                  0.321 |         0.4   |              0.625 |           0.075 |              0.05  |               0.05  |            0.275 |                    -9.811 |                 -9.119 |  40 |
| pr_k5_plain_warnsys   |                   0.981 |                  0.479 |         0.325 |              0.55  |           0.025 |              0.025 |               0.025 |            0.025 |                   -11.739 |                -10.758 |  40 |
| pr_k5_plain_nosys     |                   0.534 |                  0.171 |         0.375 |              0.55  |           0.025 |              0.05  |               0.05  |            0     |                   -11.245 |                -10.711 |  40 |
| pu_k1_history         |                   0.765 |                  0.095 |         0.375 |              0.55  |           0.025 |              0.075 |               0.075 |            0     |                   -10.497 |                 -9.732 |  40 |
| pu_k5_history         |                  -0.384 |                 -0.775 |         0.55  |              0.65  |           0.05  |              0.15  |               0.125 |            0     |                   -10.796 |                -11.18  |  40 |
| pu_k5_history_ctrlsys |                  -0.636 |                 -0.963 |         0.55  |              0.65  |           0.05  |              0.15  |               0.125 |            0     |                    -9.764 |                -10.4   |  40 |
| auth_trust            |                  -2.664 |                 -2.165 |         0.625 |              0.8   |           0.075 |              0.2   |               0.15  |            0.025 |                    -9.471 |                -12.136 |  40 |
| auth_bias             |                  -2.303 |                 -2.493 |         0.575 |              0.775 |           0.025 |              0.1   |               0.1   |            0     |                   -10.779 |                -13.082 |  40 |
| auth_bias_ctrlsys     |                  -3.321 |                 -3.207 |         0.725 |              0.9   |           0     |              0.3   |               0.3   |            0.125 |                    -9.337 |                -12.658 |  40 |
| ngram_orig            |                  -1.891 |                 -1.863 |         0.6   |              0.75  |           0     |              0.175 |               0.175 |            0     |                   -12.376 |                -14.267 |  40 |
| ngram_matched         |                  -2.402 |                 -2.112 |         0.625 |              0.8   |           0.025 |              0.125 |               0.125 |            0     |                   -11.44  |                -13.842 |  40 |
| ngram_matched_ctrlsys |                  -2.218 |                 -2.004 |         0.6   |              0.775 |           0     |              0.375 |               0.375 |            0.05  |                   -11.89  |                -14.108 |  40 |

Dose–response (identical plain repeats, participant frame):

|   k |   margin_first_prefixed |   belief_flip |   greedy_has_wrong |   greedy_refusal |   n |
|----:|------------------------:|--------------:|-------------------:|-----------------:|----:|
|   0 |                   2.744 |         0.225 |              0.025 |            0.375 |  40 |
|   1 |                   0.806 |         0.375 |              0.025 |            0.25  |  40 |
|   2 |                   0.236 |         0.4   |              0     |            0.175 |  40 |
|   3 |                   0.367 |         0.375 |              0.075 |            0.325 |  40 |
|   5 |                   0.194 |         0.35  |              0.1   |            0.425 |  40 |
|   8 |                  -0.497 |         0.45  |              0.125 |            0.6   |  40 |

Paired contrasts (a − b), bootstrap 95% CI over items:

| contrast                                                               |   n_items |   d_margin_first_prefixed |   d_margin_first_prefixed_lo |   d_margin_first_prefixed_hi |   d_belief_flip |   d_greedy_has_wrong |   d_greedy_has_gt |   d_greedy_refusal |
|:-----------------------------------------------------------------------|----------:|--------------------------:|-----------------------------:|-----------------------------:|----------------:|---------------------:|------------------:|-------------------:|
| frame only (k=0) vs control                                            |        40 |                     0.524 |                       -0.056 |                        1.086 |          -0.025 |                0     |            -0.025 |              0.375 |
| filler lines vs frame only                                             |        40 |                    -0.273 |                       -0.844 |                        0.255 |           0     |                0.025 |             0.05  |             -0.225 |
| correct peers vs frame only                                            |        40 |                     3.297 |                        1.97  |                        4.612 |          -0.15  |               -0.025 |             0.3   |             -0.225 |
| 1 repeat vs frame only                                                 |        40 |                    -1.938 |                       -3.322 |                       -0.648 |           0.15  |                0     |             0.025 |             -0.125 |
| 5 repeats vs 1 repeat                                                  |        40 |                    -0.612 |                       -1.342 |                        0.169 |          -0.025 |                0.075 |             0.025 |              0.175 |
| 8 repeats vs 5 repeats                                                 |        40 |                    -0.691 |                       -1.101 |                       -0.333 |           0.1   |                0.025 |            -0.05  |              0.175 |
| QD (stated once) vs 5 repeats                                          |        40 |                     0.498 |                       -0.266 |                        1.225 |           0.05  |               -0.05  |             0     |             -0.15  |
| diverse (no consensus) vs 5 repeats                                    |        40 |                    -0.699 |                       -1.993 |                        0.568 |           0.15  |               -0.1   |            -0.05  |              0.05  |
| DA (4+1) vs 5 repeats                                                  |        40 |                    -1.121 |                       -1.855 |                       -0.428 |           0.1   |               -0.05  |            -0.025 |             -0.05  |
| confident (varied) vs plain (identical)                                |        40 |                    -0.911 |                       -1.769 |                       -0.153 |           0.125 |                0.075 |            -0.075 |              0.125 |
| uncertain vs plain                                                     |        40 |                    -0.323 |                       -0.936 |                        0.265 |           0.1   |               -0.025 |             0.05  |             -0.325 |
| warning sys vs control sys (participant frame)                         |        40 |                     0.787 |                        0.309 |                        1.243 |          -0.025 |               -0.075 |            -0.05  |             -0.4   |
| no sys vs control sys (participant frame)                              |        40 |                     0.34  |                       -0.082 |                        0.764 |           0.025 |               -0.05  |            -0.05  |             -0.425 |
| prior-users frame vs participant frame (both control sys, 5 identical) |        40 |                    -0.83  |                       -1.893 |                        0.246 |           0.2   |                0.05  |            -0.025 |             -0.425 |
| warning sys vs control sys (prior-users frame)                         |        40 |                     0.251 |                       -0.31  |                        0.768 |           0     |                0     |             0     |              0     |
| 5 vs 1 repeats (prior-users frame, warning sys)                        |        40 |                    -1.149 |                       -2.065 |                       -0.241 |           0.175 |                0.075 |             0.025 |              0     |
| authority trust (1 claim) vs 5 repeats participant                     |        40 |                    -2.858 |                       -4.762 |                       -1.029 |           0.275 |                0.1   |             0     |             -0.4   |
| authority bias: warning vs control sys                                 |        40 |                     1.018 |                        0.349 |                        1.693 |          -0.15  |               -0.2   |             0.025 |             -0.125 |
| ngram matched vs participant frame (both no sys)                       |        40 |                    -2.936 |                       -4.747 |                       -1.119 |           0.25  |                0.075 |             0     |              0     |
| ngram orig vs ngram matched (instruction wording)                      |        40 |                     0.511 |                       -0.086 |                        1.14  |          -0.025 |                0.05  |            -0.025 |              0     |
| ngram matched+ctrl sys vs participant frame+ctrl sys                   |        40 |                    -2.412 |                       -4.284 |                       -0.512 |           0.25  |                0.275 |            -0.075 |             -0.375 |

Suppression proxy — among greedy-wrong outputs, fraction whose forced-answer belief still favours GT:

| condition             |   n_greedy_wrong |   frac_belief_still_gt |
|:----------------------|-----------------:|-----------------------:|
| auth_bias             |                4 |                  0     |
| auth_bias_ctrlsys     |               12 |                  0.25  |
| auth_trust            |                6 |                  0.333 |
| control               |                1 |                  0     |
| ngram_matched         |                5 |                  0.4   |
| ngram_matched_ctrlsys |               15 |                  0.133 |
| ngram_orig            |                7 |                  0.286 |
| pr_k0_plain           |                1 |                  1     |
| pr_k1_plain           |                1 |                  1     |
| pr_k3_plain           |                3 |                  0.333 |
| pr_k5_confident       |                7 |                  0.286 |
| pr_k5_da              |                2 |                  0.5   |
| pr_k5_filler          |                2 |                  1     |
| pr_k5_plain           |                4 |                  0.5   |
| pr_k5_plain_nosys     |                2 |                  0.5   |
| pr_k5_plain_warnsys   |                1 |                  0     |
| pr_k5_uncertain       |                3 |                  0.333 |
| pr_k8_plain           |                5 |                  0.4   |
| pr_qd                 |                2 |                  1     |
| pu_k1_history         |                3 |                  0.333 |
| pu_k5_history         |                5 |                  0.2   |
| pu_k5_history_ctrlsys |                5 |                  0.4   |

## instruct_sft

| condition             |   margin_first_prefixed |   margin_mean_prefixed |   belief_flip |   belief_flip_mean |   greedy_has_gt |   greedy_has_wrong |   greedy_wrong_only |   greedy_refusal |   lp_first_wrong_prefixed |   lp_first_gt_prefixed |   n |
|:----------------------|------------------------:|-----------------------:|--------------:|-------------------:|----------------:|-------------------:|--------------------:|-----------------:|--------------------------:|-----------------------:|----:|
| control               |                   1.795 |                  1.264 |         0.2   |              0.375 |           0.075 |              0.05  |               0.05  |            0     |                   -11.208 |                 -9.413 |  40 |
| pr_k0_plain           |                   1.743 |                  1.247 |         0.2   |              0.325 |           0.05  |              0.075 |               0.075 |            0.325 |                   -11.447 |                 -9.704 |  40 |
| pr_k5_filler          |                   1.539 |                  1.111 |         0.25  |              0.425 |           0.025 |              0.05  |               0.05  |            0.15  |                   -11.519 |                 -9.98  |  40 |
| pr_k5_correct         |                   6.196 |                  4.401 |         0.05  |              0.025 |           0.625 |              0.025 |               0.025 |            0.1   |                   -12.743 |                 -6.547 |  40 |
| pr_k1_plain           |                  -1.837 |                 -1.73  |         0.575 |              0.75  |           0.1   |              0.175 |               0.175 |            0.2   |                    -7.828 |                 -9.665 |  40 |
| pr_k2_plain           |                  -2.476 |                 -2.232 |         0.6   |              0.85  |           0     |              0.375 |               0.375 |            0.175 |                    -7.034 |                 -9.51  |  40 |
| pr_k3_plain           |                  -2.542 |                 -2.225 |         0.625 |              0.85  |           0     |              0.375 |               0.375 |            0.225 |                    -6.868 |                 -9.41  |  40 |
| pr_k5_plain           |                  -2.847 |                 -2.404 |         0.65  |              0.85  |           0     |              0.425 |               0.425 |            0.325 |                    -6.865 |                 -9.712 |  40 |
| pr_k8_plain           |                  -3.1   |                 -2.547 |         0.675 |              0.85  |           0     |              0.425 |               0.425 |            0.25  |                    -6.943 |                -10.042 |  40 |
| pr_k5_confident       |                  -3.787 |                 -2.884 |         0.75  |              0.975 |           0     |              0.55  |               0.55  |            0.25  |                    -5.887 |                 -9.674 |  40 |
| pr_k5_uncertain       |                  -2.569 |                 -2.23  |         0.675 |              0.85  |           0.05  |              0.5   |               0.5   |            0.375 |                    -6.73  |                 -9.299 |  40 |
| pr_k5_diverse         |                  -2.524 |                 -2.273 |         0.6   |              0.8   |           0     |              0.15  |               0.15  |            0.5   |                    -9.026 |                -11.55  |  40 |
| pr_k5_da              |                  -3.404 |                 -2.814 |         0.7   |              0.875 |           0.025 |              0.325 |               0.325 |            0.275 |                    -7.035 |                -10.439 |  40 |
| pr_qd                 |                  -1.376 |                 -1.304 |         0.475 |              0.725 |           0.025 |              0.325 |               0.325 |            0.35  |                    -7.557 |                 -8.933 |  40 |
| pr_k5_plain_warnsys   |                  -2.319 |                 -1.971 |         0.6   |              0.825 |           0.05  |              0.25  |               0.25  |            0     |                    -7.015 |                 -9.334 |  40 |
| pr_k5_plain_nosys     |                  -2.675 |                 -2.306 |         0.7   |              0.85  |           0.025 |              0.3   |               0.3   |            0     |                    -7.177 |                 -9.851 |  40 |
| pu_k1_history         |                  -2.164 |                 -1.794 |         0.6   |              0.8   |           0.025 |              0.4   |               0.4   |            0     |                    -7.274 |                 -9.438 |  40 |
| pu_k5_history         |                  -3.447 |                 -2.705 |         0.7   |              0.875 |           0.05  |              0.575 |               0.575 |            0     |                    -7.039 |                -10.486 |  40 |
| pu_k5_history_ctrlsys |                  -3.415 |                 -2.78  |         0.725 |              0.9   |           0     |              0.625 |               0.625 |            0     |                    -7.032 |                -10.446 |  40 |
| auth_trust            |                  -3.109 |                 -2.319 |         0.725 |              0.9   |           0.025 |              0.3   |               0.275 |            0     |                    -6.792 |                 -9.901 |  40 |
| auth_bias             |                  -2.588 |                 -2.247 |         0.725 |              0.9   |           0.025 |              0.25  |               0.25  |            0     |                    -7.153 |                 -9.741 |  40 |
| auth_bias_ctrlsys     |                  -2.955 |                 -2.534 |         0.75  |              0.95  |           0.025 |              0.375 |               0.375 |            0     |                    -6.736 |                 -9.691 |  40 |
| ngram_orig            |                  -3.415 |                 -2.556 |         0.625 |              0.875 |           0     |              0.325 |               0.325 |            0     |                    -7.916 |                -11.331 |  40 |
| ngram_matched         |                  -3.438 |                 -2.588 |         0.725 |              0.925 |           0     |              0.575 |               0.575 |            0     |                    -7.742 |                -11.18  |  40 |
| ngram_matched_ctrlsys |                  -3.531 |                 -2.673 |         0.7   |              0.925 |           0     |              0.45  |               0.45  |            0     |                    -8.025 |                -11.556 |  40 |

Dose–response (identical plain repeats, participant frame):

|   k |   margin_first_prefixed |   belief_flip |   greedy_has_wrong |   greedy_refusal |   n |
|----:|------------------------:|--------------:|-------------------:|-----------------:|----:|
|   0 |                   1.743 |         0.2   |              0.075 |            0.325 |  40 |
|   1 |                  -1.837 |         0.575 |              0.175 |            0.2   |  40 |
|   2 |                  -2.476 |         0.6   |              0.375 |            0.175 |  40 |
|   3 |                  -2.542 |         0.625 |              0.375 |            0.225 |  40 |
|   5 |                  -2.847 |         0.65  |              0.425 |            0.325 |  40 |
|   8 |                  -3.1   |         0.675 |              0.425 |            0.25  |  40 |

Paired contrasts (a − b), bootstrap 95% CI over items:

| contrast                                                               |   n_items |   d_margin_first_prefixed |   d_margin_first_prefixed_lo |   d_margin_first_prefixed_hi |   d_belief_flip |   d_greedy_has_wrong |   d_greedy_has_gt |   d_greedy_refusal |
|:-----------------------------------------------------------------------|----------:|--------------------------:|-----------------------------:|-----------------------------:|----------------:|---------------------:|------------------:|-------------------:|
| frame only (k=0) vs control                                            |        40 |                    -0.052 |                       -0.48  |                        0.385 |           0     |                0.025 |            -0.025 |              0.325 |
| filler lines vs frame only                                             |        40 |                    -0.203 |                       -0.624 |                        0.198 |           0.05  |               -0.025 |            -0.025 |             -0.175 |
| correct peers vs frame only                                            |        40 |                     4.454 |                        3.31  |                        5.617 |          -0.15  |               -0.05  |             0.575 |             -0.225 |
| 1 repeat vs frame only                                                 |        40 |                    -3.58  |                       -4.829 |                       -2.364 |           0.375 |                0.1   |             0.05  |             -0.125 |
| 5 repeats vs 1 repeat                                                  |        40 |                    -1.01  |                       -1.59  |                       -0.426 |           0.075 |                0.25  |            -0.1   |              0.125 |
| 8 repeats vs 5 repeats                                                 |        40 |                    -0.253 |                       -0.438 |                       -0.074 |           0.025 |                0     |             0     |             -0.075 |
| QD (stated once) vs 5 repeats                                          |        40 |                     1.471 |                        0.798 |                        2.09  |          -0.175 |               -0.1   |             0.025 |              0.025 |
| diverse (no consensus) vs 5 repeats                                    |        40 |                     0.323 |                       -0.464 |                        1.16  |          -0.05  |               -0.275 |             0     |              0.175 |
| DA (4+1) vs 5 repeats                                                  |        40 |                    -0.557 |                       -0.932 |                       -0.191 |           0.05  |               -0.1   |             0.025 |             -0.05  |
| confident (varied) vs plain (identical)                                |        40 |                    -0.94  |                       -1.524 |                       -0.408 |           0.1   |                0.125 |             0     |             -0.075 |
| uncertain vs plain                                                     |        40 |                     0.278 |                       -0.195 |                        0.733 |           0.025 |                0.075 |             0.05  |              0.05  |
| warning sys vs control sys (participant frame)                         |        40 |                     0.528 |                        0.203 |                        0.849 |          -0.05  |               -0.175 |             0.05  |             -0.325 |
| no sys vs control sys (participant frame)                              |        40 |                     0.172 |                       -0.208 |                        0.521 |           0.05  |               -0.125 |             0.025 |             -0.325 |
| prior-users frame vs participant frame (both control sys, 5 identical) |        40 |                    -0.568 |                       -1.28  |                        0.137 |           0.075 |                0.2   |             0     |             -0.325 |
| warning sys vs control sys (prior-users frame)                         |        40 |                    -0.033 |                       -0.305 |                        0.219 |          -0.025 |               -0.05  |             0.05  |              0     |
| 5 vs 1 repeats (prior-users frame, warning sys)                        |        40 |                    -1.284 |                       -1.844 |                       -0.758 |           0.1   |                0.175 |             0.025 |              0     |
| authority trust (1 claim) vs 5 repeats participant                     |        40 |                    -0.262 |                       -1.083 |                        0.578 |           0.075 |               -0.125 |             0.025 |             -0.325 |
| authority bias: warning vs control sys                                 |        40 |                     0.367 |                        0.03  |                        0.702 |          -0.025 |               -0.125 |             0     |              0     |
| ngram matched vs participant frame (both no sys)                       |        40 |                    -0.763 |                       -1.797 |                        0.152 |           0.025 |                0.275 |            -0.025 |              0     |
| ngram orig vs ngram matched (instruction wording)                      |        40 |                     0.023 |                       -0.346 |                        0.391 |          -0.1   |               -0.25  |             0     |              0     |
| ngram matched+ctrl sys vs participant frame+ctrl sys                   |        40 |                    -0.684 |                       -1.68  |                        0.258 |           0.05  |                0.025 |             0     |             -0.325 |

Suppression proxy — among greedy-wrong outputs, fraction whose forced-answer belief still favours GT:

| condition             |   n_greedy_wrong |   frac_belief_still_gt |
|:----------------------|-----------------:|-----------------------:|
| auth_bias             |               10 |                  0.3   |
| auth_bias_ctrlsys     |               15 |                  0.267 |
| auth_trust            |               11 |                  0.273 |
| control               |                2 |                  1     |
| ngram_matched         |               23 |                  0.261 |
| ngram_matched_ctrlsys |               18 |                  0.278 |
| ngram_orig            |               13 |                  0.308 |
| pr_k0_plain           |                3 |                  1     |
| pr_k1_plain           |                7 |                  0.286 |
| pr_k2_plain           |               15 |                  0.467 |
| pr_k3_plain           |               15 |                  0.267 |
| pr_k5_confident       |               22 |                  0.273 |
| pr_k5_correct         |                1 |                  1     |
| pr_k5_da              |               13 |                  0.385 |
| pr_k5_diverse         |                6 |                  0.5   |
| pr_k5_filler          |                2 |                  0.5   |
| pr_k5_plain           |               17 |                  0.294 |
| pr_k5_plain_nosys     |               12 |                  0.333 |
| pr_k5_plain_warnsys   |               10 |                  0.4   |
| pr_k5_uncertain       |               20 |                  0.35  |
| pr_k8_plain           |               17 |                  0.235 |
| pr_qd                 |               13 |                  0.615 |
| pu_k1_history         |               16 |                  0.25  |
| pu_k5_history         |               23 |                  0.304 |
| pu_k5_history_ctrlsys |               25 |                  0.24  |

