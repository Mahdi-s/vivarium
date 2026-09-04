# Dolci-Instruct-DPO: what the preference data reward at the policy level

N pairs = 259,922; same-model pairs = 0; preference types = {'llm_judged': 124980, 'delta_learning': 124942, 'multiturn_self_talk': 5000, 'multiturn_synthetic_context': 5000}

Features are regex hits on the final assistant turn (chosen vs rejected). `cliffs_delta_rej_minus_cho` > 0 means the rejected response has the feature more often (DPO penalises it); < 0 means DPO *favours* it. Sign test on within-pair differences.

## all (n=259,922)

| feature    |   chosen_rate |   rejected_rate |   chosen_over_rejected_ratio |   cliffs_delta_rej_minus_cho |   pairs_rej_only |   pairs_cho_only |   sign_test_p |
|:-----------|--------------:|----------------:|-----------------------------:|-----------------------------:|-----------------:|-----------------:|--------------:|
| hedge      |        0.0011 |          0.0018 |                       0.5903 |                       0.0008 |              421 |              226 |        0      |
| refusal    |        0.0301 |          0.0156 |                       1.9335 |                      -0.0145 |             2254 |             6033 |        0      |
| pushback   |        0.0228 |          0.024  |                       0.9492 |                       0.0012 |             5617 |             5300 |        0.0025 |
| agree_user |        0.017  |          0.0061 |                       2.7906 |                      -0.0109 |             1065 |             3896 |        0      |
| definite   |        0.1176 |          0.1102 |                       1.0678 |                      -0.0075 |            14012 |            15953 |        0      |
| len        |     2136.49   |       1947.38   |                       1.0971 |                      -0.109  |            83847 |           171879 |        0      |

## pref=delta_learning (n=124,942)

| feature    |   chosen_rate |   rejected_rate |   chosen_over_rejected_ratio |   cliffs_delta_rej_minus_cho |   pairs_rej_only |   pairs_cho_only |   sign_test_p |
|:-----------|--------------:|----------------:|-----------------------------:|-----------------------------:|-----------------:|-----------------:|--------------:|
| hedge      |        0.0006 |          0.0017 |                       0.3581 |                       0.0011 |              196 |               58 |         0     |
| refusal    |        0.04   |          0.0083 |                       4.83   |                      -0.0317 |              438 |             4402 |         0     |
| pushback   |        0.0174 |          0.0208 |                       0.8343 |                       0.0034 |             2349 |             1918 |         0     |
| agree_user |        0.0148 |          0.0049 |                       3.0462 |                      -0.0099 |              344 |             1584 |         0     |
| definite   |        0.1167 |          0.1151 |                       1.0132 |                      -0.0015 |             5308 |             5498 |         0.069 |
| len        |     1867.44   |       1698.94   |                       1.0992 |                      -0.1203 |            32133 |            89755 |         0     |

## pref=llm_judged (n=124,980)

| feature    |   chosen_rate |   rejected_rate |   chosen_over_rejected_ratio |   cliffs_delta_rej_minus_cho |   pairs_rej_only |   pairs_cho_only |   sign_test_p |
|:-----------|--------------:|----------------:|-----------------------------:|-----------------------------:|-----------------:|-----------------:|--------------:|
| hedge      |        0.0015 |          0.002  |                       0.7439 |                       0.0005 |              216 |              153 |        0.0012 |
| refusal    |        0.022  |          0.0236 |                       0.9339 |                       0.0016 |             1770 |             1575 |        0.0008 |
| pushback   |        0.0295 |          0.0287 |                       1.0253 |                      -0.0007 |             3230 |             3321 |        0.2662 |
| agree_user |        0.0203 |          0.0078 |                       2.6165 |                      -0.0125 |              718 |             2286 |        0      |
| definite   |        0.1239 |          0.1111 |                       1.1152 |                      -0.0128 |             8573 |            10172 |        0      |
| len        |     2511.68   |       2272.57   |                       1.1052 |                      -0.0976 |            48560 |            75669 |        0      |

## pref=multiturn_self_talk (n=5,000)

| feature    |   chosen_rate |   rejected_rate |   chosen_over_rejected_ratio |   cliffs_delta_rej_minus_cho |   pairs_rej_only |   pairs_cho_only |   sign_test_p |
|:-----------|--------------:|----------------:|-----------------------------:|-----------------------------:|-----------------:|-----------------:|--------------:|
| hedge      |        0.0016 |          0.0002 |                       8      |                      -0.0014 |                0 |                7 |        0.0156 |
| refusal    |        0.0096 |          0.0028 |                       3.4286 |                      -0.0068 |                2 |               36 |        0      |
| pushback   |        0.0068 |          0.0032 |                       2.125  |                      -0.0036 |               11 |               29 |        0.0064 |
| agree_user |        0.0042 |          0.0004 |                      10.5    |                      -0.0038 |                2 |               21 |        0.0001 |
| definite   |        0.0126 |          0.0124 |                       1.0161 |                      -0.0002 |               27 |               28 |        1      |
| len        |      910.099  |        940.138  |                       0.968  |                      -0.0899 |             1303 |             3546 |        0      |

## pref=multiturn_synthetic_context (n=5,000)

| feature    |   chosen_rate |   rejected_rate |   chosen_over_rejected_ratio |   cliffs_delta_rej_minus_cho |   pairs_rej_only |   pairs_cho_only |   sign_test_p |
|:-----------|--------------:|----------------:|-----------------------------:|-----------------------------:|-----------------:|-----------------:|--------------:|
| hedge      |        0.0026 |          0.0028 |                       0.9286 |                       0.0002 |                9 |                8 |        1      |
| refusal    |        0.0054 |          0.0102 |                       0.5294 |                       0.0048 |               44 |               20 |        0.0037 |
| pushback   |        0.0068 |          0.0058 |                       1.1724 |                      -0.001  |               27 |               32 |        0.6029 |
| agree_user |        0.0014 |          0.0006 |                       2.3333 |                      -0.0008 |                1 |                5 |        0.2188 |
| definite   |        0.0908 |          0.0606 |                       1.4983 |                      -0.0302 |              104 |              255 |        0      |
| len        |      707.843  |       1034.17   |                       0.6845 |                      -0.0508 |             1851 |             2909 |        0      |

## same_model_pairs (n=0)

| feature    |   chosen_rate |   rejected_rate |   chosen_over_rejected_ratio |   cliffs_delta_rej_minus_cho |   pairs_rej_only |   pairs_cho_only |   sign_test_p |
|:-----------|--------------:|----------------:|-----------------------------:|-----------------------------:|-----------------:|-----------------:|--------------:|
| hedge      |           nan |             nan |                          nan |                          nan |                0 |                0 |             1 |
| refusal    |           nan |             nan |                          nan |                          nan |                0 |                0 |             1 |
| pushback   |           nan |             nan |                          nan |                          nan |                0 |                0 |             1 |
| agree_user |           nan |             nan |                          nan |                          nan |                0 |                0 |             1 |
| definite   |           nan |             nan |                          nan |                          nan |                0 |                0 |             1 |
| len        |           nan |             nan |                          nan |                          nan |                0 |                0 |             1 |

## different_model_pairs (n=259,922)

| feature    |   chosen_rate |   rejected_rate |   chosen_over_rejected_ratio |   cliffs_delta_rej_minus_cho |   pairs_rej_only |   pairs_cho_only |   sign_test_p |
|:-----------|--------------:|----------------:|-----------------------------:|-----------------------------:|-----------------:|-----------------:|--------------:|
| hedge      |        0.0011 |          0.0018 |                       0.5903 |                       0.0008 |              421 |              226 |        0      |
| refusal    |        0.0301 |          0.0156 |                       1.9335 |                      -0.0145 |             2254 |             6033 |        0      |
| pushback   |        0.0228 |          0.024  |                       0.9492 |                       0.0012 |             5617 |             5300 |        0.0025 |
| agree_user |        0.017  |          0.0061 |                       2.7906 |                      -0.0109 |             1065 |             3896 |        0      |
| definite   |        0.1176 |          0.1102 |                       1.0678 |                      -0.0075 |            14012 |            15953 |        0      |
| len        |     2136.49   |       1947.38   |                       1.0971 |                      -0.109  |            83847 |           171879 |        0      |

## Model-pair confound

Chosen models (top 8):

| chosen_model              |      n |   hedge |   pushback |   definite |
|:--------------------------|-------:|--------:|-----------:|-----------:|
| qwen3-no_reasoning-32b    | 136122 |   0.001 |      0.017 |      0.118 |
| gpt-4.1-2025-04-14        |   7809 |   0     |      0.025 |      0.118 |
| qwen3-no_reasoning-1.7b   |   7757 |   0.001 |      0.027 |      0.304 |
| gpt-120b                  |   7750 |   0     |      0.016 |      0.024 |
| olmo2-13b                 |   7108 |   0.003 |      0.025 |      0.055 |
| qwen3-no_reasoning-30b-3a |   6856 |   0.001 |      0.068 |      0.267 |
| gpt-20b                   |   6836 |   0     |      0.014 |      0.015 |
| yi-34b                    |   6650 |   0.004 |      0.025 |      0.025 |

Rejected models (top 8):

| rejected_model          |      n |   hedge |   pushback |   definite |
|:------------------------|-------:|--------:|-----------:|-----------:|
| qwen3-no_reasoning-0.6b | 149385 |   0.002 |      0.021 |      0.119 |
| olmo2-1b                |  20901 |   0.001 |      0.024 |      0.122 |
| olmo2-7b                |  12529 |   0.001 |      0.023 |      0.152 |
| yi-9b                   |  12132 |   0.001 |      0.039 |      0.015 |
| qwen3-no_reasoning-1.7b |  12022 |   0.005 |      0.051 |      0.18  |
| phi4-mini-instruct      |  10315 |   0.004 |      0.022 |      0.021 |
| olmo2-13b               |   9743 |   0.001 |      0.018 |      0.159 |
| yi-34b                  |   9064 |   0.002 |      0.032 |      0.013 |
