# Cross-family 4-state abandonment (control-correct items), Wilson 95% CI

## Control accuracy

| variant              |   temperature |   n |   ctrl_acc |   ctrl_refusal |
|:---------------------|--------------:|----:|-----------:|---------------:|
| claude_sonnet_4      |           0   | 400 |      0.65  |          0.102 |
| claude_sonnet_4      |           0.6 | 400 |      0.68  |          0.098 |
| gemini_25_flash_lite |           0   | 400 |      0.612 |          0.08  |
| gemini_25_flash_lite |           0.6 | 400 |      0.598 |          0.07  |
| gpt4o_mini           |           0   | 400 |      0.6   |          0.055 |
| gpt4o_mini           |           0.6 | 400 |      0.612 |          0.062 |
| gpt_oss_20b          |           0   | 400 |      0.52  |          0.138 |
| gpt_oss_20b          |           0.6 | 400 |      0.618 |          0.092 |
| grok_41_fast         |           0   | 400 |      0.638 |          0.135 |
| grok_41_fast         |           0.6 | 400 |      0.645 |          0.13  |
| instruct_32b         |           0   | 400 |      0.592 |          0.128 |
| instruct_32b         |           0.6 | 400 |      0.612 |          0.115 |
| llama31_70b_instruct |           0   | 400 |      0.62  |          0.06  |
| llama31_70b_instruct |           0.6 | 400 |      0.625 |          0.06  |
| llama3_8b_instruct   |           0   | 400 |      0.385 |          0.255 |
| llama3_8b_instruct   |           0.6 | 400 |      0.362 |          0.252 |
| llama4_maverick      |           0   | 400 |      0.678 |          0.025 |
| llama4_maverick      |           0.6 | 400 |      0.645 |          0.022 |
| think                |           0   | 404 |      0.512 |          0.131 |
| think_32b            |           0   | 400 |      0.63  |          0.115 |
| think_32b            |           0.6 | 400 |      0.615 |          0.12  |
| think_7b             |           0.6 | 400 |      0.47  |          0.202 |

## T=0.0: P(abandon | control correct)

| variant              |   asch_zhu_naked_unanimous_confident |   asch_zhu_unanimous_confident |   authoritative_bias |   authority_trust |   ngram_sequence_baseline |   ngram_sequence_matched_baseline |
|:---------------------|-------------------------------------:|-------------------------------:|---------------------:|------------------:|--------------------------:|----------------------------------:|
| claude_sonnet_4      |                              nan     |                          0.096 |                0.127 |             0.108 |                   nan     |                           nan     |
| gemini_25_flash_lite |                              nan     |                          0.294 |                0.188 |             0.196 |                   nan     |                           nan     |
| gpt4o_mini           |                              nan     |                          0.429 |                0.138 |             0.142 |                   nan     |                           nan     |
| gpt_oss_20b          |                              nan     |                          0.202 |                0.308 |             0.202 |                   nan     |                           nan     |
| grok_41_fast         |                              nan     |                          0.122 |                0.082 |             0.11  |                   nan     |                           nan     |
| instruct_32b         |                                0.325 |                          0.717 |                0.16  |             0.241 |                     0.515 |                           nan     |
| llama31_70b_instruct |                                0.403 |                          0.891 |                0.262 |             0.19  |                     0.48  |                             0.371 |
| llama3_8b_instruct   |                              nan     |                          0.974 |                0.227 |             0.429 |                   nan     |                           nan     |
| llama4_maverick      |                              nan     |                          0.28  |                0.185 |             0.133 |                   nan     |                           nan     |
| think                |                              nan     |                          0.259 |                0.205 |             0.254 |                   nan     |                           nan     |
| think_32b            |                              nan     |                          0.183 |                0.131 |             0.167 |                   nan     |                           nan     |

Destination = target_wrong share of abandonment (p_to_target / p_abandon):

| variant              |   asch_zhu_naked_unanimous_confident |   asch_zhu_unanimous_confident |   authoritative_bias |   authority_trust |   ngram_sequence_baseline |   ngram_sequence_matched_baseline |
|:---------------------|-------------------------------------:|-------------------------------:|---------------------:|------------------:|--------------------------:|----------------------------------:|
| claude_sonnet_4      |                               nan    |                           0.24 |                 0.18 |              0.25 |                    nan    |                            nan    |
| gemini_25_flash_lite |                               nan    |                           0.5  |                 0.33 |              0.42 |                    nan    |                            nan    |
| gpt4o_mini           |                               nan    |                           0.71 |                 0.3  |              0.53 |                    nan    |                            nan    |
| gpt_oss_20b          |                               nan    |                           0.36 |                 0.48 |              0.5  |                    nan    |                            nan    |
| grok_41_fast         |                               nan    |                           0.26 |                 0.43 |              0.54 |                    nan    |                            nan    |
| instruct_32b         |                                 0.73 |                           0.21 |                 0.32 |              0.28 |                      0.73 |                            nan    |
| llama31_70b_instruct |                                 0.58 |                           0.04 |                 0.52 |              0.45 |                      0.63 |                              0.54 |
| llama3_8b_instruct   |                               nan    |                           0.24 |                 0.26 |              0.26 |                    nan    |                            nan    |
| llama4_maverick      |                               nan    |                           0.58 |                 0.56 |              0.5  |                    nan    |                            nan    |
| think                |                               nan    |                           0.66 |                 0.48 |              0.54 |                    nan    |                            nan    |
| think_32b            |                               nan    |                           0.3  |                 0.42 |              0.31 |                    nan    |                            nan    |

Refusal destination:

| variant              |   asch_zhu_naked_unanimous_confident |   asch_zhu_unanimous_confident |   authoritative_bias |   authority_trust |   ngram_sequence_baseline |   ngram_sequence_matched_baseline |
|:---------------------|-------------------------------------:|-------------------------------:|---------------------:|------------------:|--------------------------:|----------------------------------:|
| claude_sonnet_4      |                              nan     |                          0.023 |                0.027 |             0.015 |                   nan     |                               nan |
| gemini_25_flash_lite |                              nan     |                          0.041 |                0.024 |             0.004 |                   nan     |                               nan |
| gpt4o_mini           |                              nan     |                          0.017 |                0     |             0.004 |                   nan     |                               nan |
| gpt_oss_20b          |                              nan     |                          0.034 |                0.014 |             0.005 |                   nan     |                               nan |
| grok_41_fast         |                              nan     |                          0.008 |                0.004 |             0.008 |                   nan     |                               nan |
| instruct_32b         |                                0.021 |                          0.502 |                0.004 |             0.122 |                     0.021 |                               nan |
| llama31_70b_instruct |                                0.02  |                          0.851 |                0.008 |             0     |                     0     |                                 0 |
| llama3_8b_instruct   |                              nan     |                          0.695 |                0     |             0.175 |                   nan     |                               nan |
| llama4_maverick      |                              nan     |                          0.066 |                0.004 |             0.004 |                   nan     |                               nan |
| think                |                              nan     |                          0.044 |                0.015 |             0.054 |                   nan     |                               nan |
| think_32b            |                              nan     |                          0.079 |                0     |             0.032 |                   nan     |                               nan |

## T=0.6: P(abandon | control correct)

| variant              |   asch_zhu_naked_unanimous_confident |   asch_zhu_unanimous_confident |   authoritative_bias |   authority_trust |   ngram_sequence_baseline |   ngram_sequence_matched_baseline |
|:---------------------|-------------------------------------:|-------------------------------:|---------------------:|------------------:|--------------------------:|----------------------------------:|
| claude_sonnet_4      |                              nan     |                          0.125 |                0.151 |             0.114 |                    nan    |                             nan   |
| gemini_25_flash_lite |                              nan     |                          0.31  |                0.18  |             0.159 |                    nan    |                             nan   |
| gpt4o_mini           |                              nan     |                          0.469 |                0.167 |             0.139 |                    nan    |                             nan   |
| gpt_oss_20b          |                              nan     |                          0.211 |                0.263 |             0.198 |                    nan    |                             nan   |
| grok_41_fast         |                              nan     |                          0.124 |                0.112 |             0.132 |                    nan    |                             nan   |
| instruct_32b         |                              nan     |                          0.767 |                0.176 |             0.233 |                    nan    |                             nan   |
| llama31_70b_instruct |                                0.388 |                          0.744 |                0.24  |             0.228 |                      0.48 |                               0.4 |
| llama3_8b_instruct   |                              nan     |                          0.945 |                0.221 |             0.393 |                    nan    |                             nan   |
| llama4_maverick      |                              nan     |                          0.26  |                0.136 |             0.132 |                    nan    |                             nan   |
| think_32b            |                              nan     |                          0.146 |                0.122 |             0.15  |                    nan    |                             nan   |
| think_7b             |                              nan     |                          0.314 |                0.309 |             0.33  |                    nan    |                             nan   |

Destination = target_wrong share of abandonment (p_to_target / p_abandon):

| variant              |   asch_zhu_naked_unanimous_confident |   asch_zhu_unanimous_confident |   authoritative_bias |   authority_trust |   ngram_sequence_baseline |   ngram_sequence_matched_baseline |
|:---------------------|-------------------------------------:|-------------------------------:|---------------------:|------------------:|--------------------------:|----------------------------------:|
| claude_sonnet_4      |                               nan    |                           0.24 |                 0.29 |              0.26 |                    nan    |                             nan   |
| gemini_25_flash_lite |                               nan    |                           0.42 |                 0.21 |              0.24 |                    nan    |                             nan   |
| gpt4o_mini           |                               nan    |                           0.71 |                 0.41 |              0.38 |                    nan    |                             nan   |
| gpt_oss_20b          |                               nan    |                           0.46 |                 0.51 |              0.61 |                    nan    |                             nan   |
| grok_41_fast         |                               nan    |                           0.38 |                 0.48 |              0.71 |                    nan    |                             nan   |
| instruct_32b         |                               nan    |                           0.24 |                 0.37 |              0.33 |                    nan    |                             nan   |
| llama31_70b_instruct |                                 0.58 |                           0.05 |                 0.48 |              0.23 |                      0.57 |                               0.5 |
| llama3_8b_instruct   |                               nan    |                           0.3  |                 0.19 |              0.33 |                    nan    |                             nan   |
| llama4_maverick      |                               nan    |                           0.63 |                 0.66 |              0.5  |                    nan    |                             nan   |
| think_32b            |                               nan    |                           0.56 |                 0.43 |              0.49 |                    nan    |                             nan   |
| think_7b             |                               nan    |                           0.27 |                 0.45 |              0.31 |                    nan    |                             nan   |

Refusal destination:

| variant              |   asch_zhu_naked_unanimous_confident |   asch_zhu_unanimous_confident |   authoritative_bias |   authority_trust |   ngram_sequence_baseline |   ngram_sequence_matched_baseline |
|:---------------------|-------------------------------------:|-------------------------------:|---------------------:|------------------:|--------------------------:|----------------------------------:|
| claude_sonnet_4      |                              nan     |                          0.011 |                0.022 |             0.033 |                       nan |                           nan     |
| gemini_25_flash_lite |                              nan     |                          0.038 |                0.038 |             0.013 |                       nan |                           nan     |
| gpt4o_mini           |                              nan     |                          0.016 |                0     |             0.004 |                       nan |                           nan     |
| gpt_oss_20b          |                              nan     |                          0.028 |                0.02  |             0.008 |                       nan |                           nan     |
| grok_41_fast         |                              nan     |                          0.008 |                0     |             0.008 |                       nan |                           nan     |
| instruct_32b         |                              nan     |                          0.543 |                0.004 |             0.118 |                       nan |                           nan     |
| llama31_70b_instruct |                                0.024 |                          0.7   |                0     |             0.012 |                         0 |                             0.008 |
| llama3_8b_instruct   |                              nan     |                          0.628 |                0.007 |             0.145 |                       nan |                           nan     |
| llama4_maverick      |                              nan     |                          0.058 |                0     |             0.012 |                       nan |                           nan     |
| think_32b            |                              nan     |                          0.037 |                0     |             0.041 |                       nan |                           nan     |
| think_7b             |                              nan     |                          0.112 |                0.037 |             0.085 |                       nan |                           nan     |

## Ablation detail (naked social vs n-gram vs matched-instruction n-gram)

| variant              |   temperature | condition                          |   n_ctrl_correct |   p_abandon |    lo |    hi |   p_to_target |   p_to_other |   p_to_refusal |
|:---------------------|--------------:|:-----------------------------------|-----------------:|------------:|------:|------:|--------------:|-------------:|---------------:|
| instruct_32b         |           0   | asch_zhu_naked_unanimous_confident |              237 |       0.325 | 0.268 | 0.387 |         0.236 |        0.068 |          0.021 |
| instruct_32b         |           0   | asch_zhu_unanimous_confident       |              237 |       0.717 | 0.657 | 0.771 |         0.152 |        0.063 |          0.502 |
| instruct_32b         |           0   | ngram_sequence_baseline            |              237 |       0.515 | 0.451 | 0.578 |         0.376 |        0.118 |          0.021 |
| instruct_32b         |           0.6 | asch_zhu_unanimous_confident       |              245 |       0.767 | 0.711 | 0.816 |         0.184 |        0.041 |          0.543 |
| llama31_70b_instruct |           0   | asch_zhu_naked_unanimous_confident |              248 |       0.403 | 0.344 | 0.465 |         0.234 |        0.149 |          0.02  |
| llama31_70b_instruct |           0   | asch_zhu_unanimous_confident       |              248 |       0.891 | 0.846 | 0.924 |         0.032 |        0.008 |          0.851 |
| llama31_70b_instruct |           0   | ngram_sequence_baseline            |              248 |       0.48  | 0.418 | 0.542 |         0.302 |        0.177 |          0     |
| llama31_70b_instruct |           0   | ngram_sequence_matched_baseline    |              248 |       0.371 | 0.313 | 0.433 |         0.202 |        0.169 |          0     |
| llama31_70b_instruct |           0.6 | asch_zhu_naked_unanimous_confident |              250 |       0.388 | 0.33  | 0.45  |         0.224 |        0.14  |          0.024 |
| llama31_70b_instruct |           0.6 | asch_zhu_unanimous_confident       |              250 |       0.744 | 0.686 | 0.794 |         0.036 |        0.008 |          0.7   |
| llama31_70b_instruct |           0.6 | ngram_sequence_baseline            |              250 |       0.48  | 0.419 | 0.542 |         0.272 |        0.208 |          0     |
| llama31_70b_instruct |           0.6 | ngram_sequence_matched_baseline    |              250 |       0.4   | 0.341 | 0.462 |         0.2   |        0.192 |          0.008 |
