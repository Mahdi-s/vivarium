# Core 4-state analysis — OLMo-3 7B family, publication item set

Trials analysed: 201540; paired pressure trials: 184740; control-correct pairs: 76768.

Outcome rule: correct (judge) → refusal (SQL or judge flag) → target_wrong (judge endorsed injected answer) → other_wrong.

## Truth abandonment among control-correct items (T=0)

P(abandon | control correct):

|                                     |   base |   instruct_sft |   instruct_dpo |   instruct |   think_sft |   think_dpo |   think |
|:------------------------------------|-------:|---------------:|---------------:|-----------:|------------:|------------:|--------:|
| qd(1 stated)                        |  0.737 |          0.747 |          0.521 |      0.656 |       0.909 |       0.923 |   0.927 |
| unan_plain(5 ident)                 |  0.869 |          0.699 |          0.535 |      0.617 |       0.894 |       0.937 |   0.94  |
| unan_neutral(5 var)                 |  0.899 |          0.836 |          0.528 |      0.602 |       0.929 |       0.946 |   0.945 |
| unan_confident(5 var)               |  0.899 |          0.842 |          0.606 |      0.828 |       0.914 |       0.928 |   0.94  |
| unan_uncertain(5 var)               |  0.949 |          0.74  |          0.408 |      0.516 |       0.914 |       0.932 |   0.931 |
| diverse(no consensus)               |  0.828 |          0.582 |          0.549 |      0.656 |       0.909 |       0.919 |   0.95  |
| da(4+1)                             |  0.919 |          0.623 |          0.493 |      0.633 |       0.934 |       0.946 |   0.959 |
| asch_hist(5 ident,warn,prior-users) |  0.636 |          0.712 |          0.507 |      0.516 |       0.424 |       0.348 |   0.404 |
| auth_bias(1,warn)                   |  0.687 |          0.61  |          0.634 |      0.578 |       0.424 |       0.33  |   0.413 |
| auth_trust(1)                       |  0.616 |          0.527 |          0.444 |      0.445 |       0.909 |       0.919 |   0.899 |
| auth_trust_da(1+alt)                |  0.657 |          0.521 |          0.493 |      0.5   |       0.944 |       0.95  |   0.927 |

Destination decomposition — instruct_sft (T=0):

|                                     |   n_ctrl_correct |   p_abandon |    lo |    hi |   p_to_target |   p_to_other |   p_to_refusal |   p_rescue |
|:------------------------------------|-----------------:|------------:|------:|------:|--------------:|-------------:|---------------:|-----------:|
| qd(1 stated)                        |              146 |       0.747 | 0.67  | 0.81  |         0.253 |        0.151 |          0.342 |      0.089 |
| unan_plain(5 ident)                 |              146 |       0.699 | 0.62  | 0.767 |         0.39  |        0.082 |          0.226 |      0.059 |
| unan_neutral(5 var)                 |              146 |       0.836 | 0.767 | 0.887 |         0.658 |        0.062 |          0.116 |      0.034 |
| unan_confident(5 var)               |              146 |       0.842 | 0.775 | 0.893 |         0.603 |        0.034 |          0.205 |      0.03  |
| unan_uncertain(5 var)               |              146 |       0.74  | 0.663 | 0.804 |         0.404 |        0.041 |          0.295 |      0.051 |
| diverse(no consensus)               |              146 |       0.582 | 0.501 | 0.659 |         0.103 |        0.185 |          0.295 |      0.093 |
| da(4+1)                             |              146 |       0.623 | 0.542 | 0.698 |         0.274 |        0.123 |          0.226 |      0.055 |
| asch_hist(5 ident,warn,prior-users) |              146 |       0.712 | 0.634 | 0.78  |         0.507 |        0.199 |          0.007 |      0.034 |
| auth_bias(1,warn)                   |              146 |       0.61  | 0.529 | 0.685 |         0.39  |        0.212 |          0.007 |      0.11  |
| auth_trust(1)                       |              146 |       0.527 | 0.447 | 0.607 |         0.281 |        0.219 |          0.027 |      0.097 |
| auth_trust_da(1+alt)                |              146 |       0.521 | 0.44  | 0.6   |         0.37  |        0.137 |          0.014 |      0.072 |

Destination decomposition — instruct_dpo (T=0):

|                                     |   n_ctrl_correct |   p_abandon |    lo |    hi |   p_to_target |   p_to_other |   p_to_refusal |   p_rescue |
|:------------------------------------|-----------------:|------------:|------:|------:|--------------:|-------------:|---------------:|-----------:|
| qd(1 stated)                        |              142 |       0.521 | 0.439 | 0.602 |         0.106 |        0.169 |          0.246 |      0.103 |
| unan_plain(5 ident)                 |              142 |       0.535 | 0.453 | 0.615 |         0.12  |        0.232 |          0.183 |      0.19  |
| unan_neutral(5 var)                 |              142 |       0.528 | 0.446 | 0.608 |         0.352 |        0.12  |          0.056 |      0.125 |
| unan_confident(5 var)               |              142 |       0.606 | 0.523 | 0.682 |         0.261 |        0.085 |          0.261 |      0.095 |
| unan_uncertain(5 var)               |              142 |       0.408 | 0.331 | 0.491 |         0.169 |        0.211 |          0.028 |      0.181 |
| diverse(no consensus)               |              142 |       0.549 | 0.467 | 0.629 |         0.07  |        0.232 |          0.246 |      0.134 |
| da(4+1)                             |              142 |       0.493 | 0.412 | 0.574 |         0.092 |        0.204 |          0.197 |      0.129 |
| asch_hist(5 ident,warn,prior-users) |              142 |       0.507 | 0.426 | 0.588 |         0.12  |        0.373 |          0.014 |      0.181 |
| auth_bias(1,warn)                   |              142 |       0.634 | 0.552 | 0.709 |         0.296 |        0.338 |          0     |      0.125 |
| auth_trust(1)                       |              142 |       0.444 | 0.364 | 0.526 |         0.183 |        0.239 |          0.021 |      0.147 |
| auth_trust_da(1+alt)                |              142 |       0.493 | 0.412 | 0.574 |         0.232 |        0.239 |          0.021 |      0.142 |

Destination decomposition — instruct (T=0):

|                                     |   n_ctrl_correct |   p_abandon |    lo |    hi |   p_to_target |   p_to_other |   p_to_refusal |   p_rescue |
|:------------------------------------|-----------------:|------------:|------:|------:|--------------:|-------------:|---------------:|-----------:|
| qd(1 stated)                        |              128 |       0.656 | 0.57  | 0.733 |         0.117 |        0.133 |          0.406 |      0.128 |
| unan_plain(5 ident)                 |              128 |       0.617 | 0.531 | 0.697 |         0.156 |        0.141 |          0.32  |      0.115 |
| unan_neutral(5 var)                 |              128 |       0.602 | 0.515 | 0.682 |         0.328 |        0.102 |          0.172 |      0.103 |
| unan_confident(5 var)               |              128 |       0.828 | 0.753 | 0.884 |         0.359 |        0.062 |          0.406 |      0.094 |
| unan_uncertain(5 var)               |              128 |       0.516 | 0.43  | 0.6   |         0.203 |        0.156 |          0.156 |      0.141 |
| diverse(no consensus)               |              128 |       0.656 | 0.57  | 0.733 |         0.078 |        0.195 |          0.383 |      0.09  |
| da(4+1)                             |              128 |       0.633 | 0.547 | 0.711 |         0.141 |        0.141 |          0.352 |      0.141 |
| asch_hist(5 ident,warn,prior-users) |              128 |       0.516 | 0.43  | 0.6   |         0.133 |        0.383 |          0     |      0.137 |
| auth_bias(1,warn)                   |              128 |       0.578 | 0.492 | 0.66  |         0.242 |        0.336 |          0     |      0.12  |
| auth_trust(1)                       |              128 |       0.445 | 0.362 | 0.532 |         0.156 |        0.25  |          0.039 |      0.115 |
| auth_trust_da(1+alt)                |              128 |       0.5   | 0.415 | 0.585 |         0.258 |        0.195 |          0.047 |      0.137 |

Destination decomposition — base (T=0):

|                                     |   n_ctrl_correct |   p_abandon |    lo |    hi |   p_to_target |   p_to_other |   p_to_refusal |   p_rescue |
|:------------------------------------|-----------------:|------------:|------:|------:|--------------:|-------------:|---------------:|-----------:|
| qd(1 stated)                        |               99 |       0.737 | 0.643 | 0.814 |         0.253 |        0.071 |          0.414 |      0.088 |
| unan_plain(5 ident)                 |               99 |       0.869 | 0.788 | 0.922 |         0.515 |        0.051 |          0.303 |      0.047 |
| unan_neutral(5 var)                 |               99 |       0.899 | 0.824 | 0.944 |         0.444 |        0.01  |          0.444 |      0.029 |
| unan_confident(5 var)               |               99 |       0.899 | 0.824 | 0.944 |         0.323 |        0.04  |          0.535 |      0.022 |
| unan_uncertain(5 var)               |               99 |       0.949 | 0.887 | 0.978 |         0.152 |        0.02  |          0.778 |      0.026 |
| diverse(no consensus)               |               99 |       0.828 | 0.742 | 0.89  |         0.172 |        0.162 |          0.495 |      0.088 |
| da(4+1)                             |               99 |       0.919 | 0.849 | 0.958 |         0.475 |        0.141 |          0.303 |      0.04  |
| asch_hist(5 ident,warn,prior-users) |               99 |       0.636 | 0.538 | 0.724 |         0.404 |        0.232 |          0     |      0.139 |
| auth_bias(1,warn)                   |               99 |       0.687 | 0.59  | 0.77  |         0.374 |        0.212 |          0.101 |      0.15  |
| auth_trust(1)                       |               99 |       0.616 | 0.518 | 0.706 |         0.273 |        0.182 |          0.162 |      0.15  |
| auth_trust_da(1+alt)                |               99 |       0.657 | 0.559 | 0.743 |         0.313 |        0.273 |          0.071 |      0.084 |

## Truth abandonment among control-correct items (pooled over 6 temperatures)

P(abandon | control correct):

|                                     |   base |   instruct_sft |   instruct_dpo |   instruct |   think_sft |   think_dpo |   think |
|:------------------------------------|-------:|---------------:|---------------:|-----------:|------------:|------------:|--------:|
| qd(1 stated)                        |  0.717 |          0.737 |          0.59  |      0.727 |       0.908 |       0.783 |   0.928 |
| unan_plain(5 ident)                 |  0.822 |          0.717 |          0.585 |      0.672 |       0.917 |       0.821 |   0.933 |
| unan_neutral(5 var)                 |  0.832 |          0.861 |          0.579 |      0.671 |       0.925 |       0.821 |   0.933 |
| unan_confident(5 var)               |  0.852 |          0.861 |          0.655 |      0.822 |       0.931 |       0.816 |   0.937 |
| unan_uncertain(5 var)               |  0.869 |          0.814 |          0.45  |      0.552 |       0.918 |       0.822 |   0.935 |
| diverse(no consensus)               |  0.756 |          0.6   |          0.585 |      0.685 |       0.926 |       0.819 |   0.943 |
| da(4+1)                             |  0.833 |          0.701 |          0.574 |      0.701 |       0.924 |       0.825 |   0.956 |
| asch_hist(5 ident,warn,prior-users) |  0.676 |          0.715 |          0.45  |      0.466 |       0.431 |       0.404 |   0.439 |
| auth_bias(1,warn)                   |  0.664 |          0.629 |          0.615 |      0.595 |       0.456 |       0.412 |   0.428 |
| auth_trust(1)                       |  0.649 |          0.558 |          0.503 |      0.488 |       0.912 |       0.784 |   0.916 |
| auth_trust_da(1+alt)                |  0.684 |          0.549 |          0.515 |      0.516 |       0.924 |       0.806 |   0.921 |

Destination decomposition — instruct_sft (pooled over 6 temperatures):

|                                     |   n_ctrl_correct |   p_abandon |    lo |    hi |   p_to_target |   p_to_other |   p_to_refusal |   p_rescue |
|:------------------------------------|-----------------:|------------:|------:|------:|--------------:|-------------:|---------------:|-----------:|
| qd(1 stated)                        |              834 |       0.737 | 0.707 | 0.766 |         0.296 |        0.135 |          0.306 |      0.089 |
| unan_plain(5 ident)                 |              834 |       0.717 | 0.686 | 0.747 |         0.362 |        0.138 |          0.217 |      0.076 |
| unan_neutral(5 var)                 |              834 |       0.861 | 0.836 | 0.883 |         0.651 |        0.058 |          0.152 |      0.045 |
| unan_confident(5 var)               |              834 |       0.861 | 0.836 | 0.883 |         0.647 |        0.041 |          0.173 |      0.031 |
| unan_uncertain(5 var)               |              834 |       0.814 | 0.786 | 0.839 |         0.381 |        0.068 |          0.365 |      0.052 |
| diverse(no consensus)               |              834 |       0.6   | 0.566 | 0.632 |         0.107 |        0.222 |          0.271 |      0.097 |
| da(4+1)                             |              834 |       0.701 | 0.67  | 0.732 |         0.287 |        0.151 |          0.264 |      0.073 |
| asch_hist(5 ident,warn,prior-users) |              834 |       0.715 | 0.683 | 0.744 |         0.507 |        0.183 |          0.024 |      0.056 |
| auth_bias(1,warn)                   |              834 |       0.629 | 0.596 | 0.662 |         0.444 |        0.181 |          0.005 |      0.091 |
| auth_trust(1)                       |              834 |       0.558 | 0.524 | 0.591 |         0.354 |        0.187 |          0.017 |      0.128 |
| auth_trust_da(1+alt)                |              834 |       0.549 | 0.515 | 0.583 |         0.369 |        0.171 |          0.008 |      0.093 |

Destination decomposition — instruct_dpo (pooled over 6 temperatures):

|                                     |   n_ctrl_correct |   p_abandon |    lo |    hi |   p_to_target |   p_to_other |   p_to_refusal |   p_rescue |
|:------------------------------------|-----------------:|------------:|------:|------:|--------------:|-------------:|---------------:|-----------:|
| qd(1 stated)                        |              862 |       0.59  | 0.557 | 0.623 |         0.088 |        0.238 |          0.265 |      0.112 |
| unan_plain(5 ident)                 |              862 |       0.585 | 0.551 | 0.617 |         0.113 |        0.23  |          0.242 |      0.143 |
| unan_neutral(5 var)                 |              862 |       0.579 | 0.546 | 0.611 |         0.36  |        0.119 |          0.1   |      0.125 |
| unan_confident(5 var)               |              862 |       0.655 | 0.623 | 0.686 |         0.266 |        0.122 |          0.268 |      0.096 |
| unan_uncertain(5 var)               |              862 |       0.45  | 0.417 | 0.483 |         0.187 |        0.195 |          0.068 |      0.164 |
| diverse(no consensus)               |              862 |       0.585 | 0.551 | 0.617 |         0.058 |        0.227 |          0.299 |      0.127 |
| da(4+1)                             |              862 |       0.574 | 0.541 | 0.607 |         0.092 |        0.23  |          0.253 |      0.125 |
| asch_hist(5 ident,warn,prior-users) |              862 |       0.45  | 0.417 | 0.483 |         0.088 |        0.357 |          0.005 |      0.17  |
| auth_bias(1,warn)                   |              862 |       0.615 | 0.582 | 0.647 |         0.268 |        0.347 |          0     |      0.112 |
| auth_trust(1)                       |              862 |       0.503 | 0.47  | 0.537 |         0.248 |        0.234 |          0.021 |      0.157 |
| auth_trust_da(1+alt)                |              862 |       0.515 | 0.482 | 0.548 |         0.242 |        0.249 |          0.023 |      0.14  |

Destination decomposition — instruct (pooled over 6 temperatures):

|                                     |   n_ctrl_correct |   p_abandon |    lo |    hi |   p_to_target |   p_to_other |   p_to_refusal |   p_rescue |
|:------------------------------------|-----------------:|------------:|------:|------:|--------------:|-------------:|---------------:|-----------:|
| qd(1 stated)                        |              802 |       0.727 | 0.695 | 0.757 |         0.127 |        0.192 |          0.408 |      0.088 |
| unan_plain(5 ident)                 |              802 |       0.672 | 0.639 | 0.704 |         0.138 |        0.203 |          0.33  |      0.099 |
| unan_neutral(5 var)                 |              802 |       0.671 | 0.638 | 0.702 |         0.359 |        0.111 |          0.201 |      0.102 |
| unan_confident(5 var)               |              802 |       0.822 | 0.794 | 0.847 |         0.367 |        0.07  |          0.385 |      0.076 |
| unan_uncertain(5 var)               |              802 |       0.552 | 0.518 | 0.586 |         0.223 |        0.146 |          0.183 |      0.126 |
| diverse(no consensus)               |              802 |       0.685 | 0.652 | 0.716 |         0.089 |        0.161 |          0.435 |      0.075 |
| da(4+1)                             |              802 |       0.701 | 0.668 | 0.731 |         0.15  |        0.17  |          0.382 |      0.105 |
| asch_hist(5 ident,warn,prior-users) |              802 |       0.466 | 0.432 | 0.501 |         0.111 |        0.355 |          0     |      0.152 |
| auth_bias(1,warn)                   |              802 |       0.595 | 0.56  | 0.628 |         0.232 |        0.359 |          0.004 |      0.123 |
| auth_trust(1)                       |              802 |       0.488 | 0.453 | 0.522 |         0.216 |        0.247 |          0.025 |      0.141 |
| auth_trust_da(1+alt)                |              802 |       0.516 | 0.482 | 0.551 |         0.239 |        0.213 |          0.064 |      0.127 |

Destination decomposition — base (pooled over 6 temperatures):

|                                     |   n_ctrl_correct |   p_abandon |    lo |    hi |   p_to_target |   p_to_other |   p_to_refusal |   p_rescue |
|:------------------------------------|-----------------:|------------:|------:|------:|--------------:|-------------:|---------------:|-----------:|
| qd(1 stated)                        |              636 |       0.717 | 0.681 | 0.751 |         0.292 |        0.2   |          0.225 |      0.109 |
| unan_plain(5 ident)                 |              636 |       0.822 | 0.791 | 0.85  |         0.514 |        0.099 |          0.209 |      0.08  |
| unan_neutral(5 var)                 |              636 |       0.832 | 0.801 | 0.859 |         0.461 |        0.097 |          0.274 |      0.069 |
| unan_confident(5 var)               |              636 |       0.852 | 0.823 | 0.878 |         0.426 |        0.124 |          0.302 |      0.069 |
| unan_uncertain(5 var)               |              636 |       0.869 | 0.841 | 0.893 |         0.179 |        0.057 |          0.634 |      0.049 |
| diverse(no consensus)               |              636 |       0.756 | 0.721 | 0.788 |         0.153 |        0.241 |          0.363 |      0.087 |
| da(4+1)                             |              636 |       0.833 | 0.802 | 0.86  |         0.447 |        0.156 |          0.231 |      0.069 |
| asch_hist(5 ident,warn,prior-users) |              636 |       0.676 | 0.639 | 0.711 |         0.431 |        0.244 |          0.002 |      0.142 |
| auth_bias(1,warn)                   |              636 |       0.664 | 0.626 | 0.699 |         0.393 |        0.219 |          0.052 |      0.129 |
| auth_trust(1)                       |              636 |       0.649 | 0.611 | 0.685 |         0.316 |        0.181 |          0.153 |      0.146 |
| auth_trust_da(1+alt)                |              636 |       0.684 | 0.647 | 0.719 |         0.291 |        0.245 |          0.148 |      0.117 |

## Pre-specified structural contrasts (T=0) — abandonment(a) − abandonment(b), item-bootstrap 95% CI, exact McNemar, Holm within variant

### base

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |        99 |        0.6162 |        0.8687 | -0.2525 | -0.3535 | -0.1515 |           5 |          30 |   0.0002 |
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |        99 |        0.6364 |        0.8687 | -0.2323 | -0.3333 | -0.1313 |           5 |          28 |   0.0007 |
| REPETITION 1 vs 5 (frame const)                                     |        99 |        0.7374 |        0.8687 | -0.1313 | -0.2323 | -0.0303 |           8 |          21 |   0.2171 |
| TONE: uncertain vs plain                                            |        99 |        0.9495 |        0.8687 |  0.0808 |  0.0101 |  0.1515 |          11 |           3 |   0.459  |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |        99 |        0.9192 |        0.8687 |  0.0505 | -0.0101 |  0.1111 |           7 |           2 |   0.6177 |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |        99 |        0.6869 |        0.6162 |  0.0707 | -0.0404 |  0.1919 |          21 |          14 |   0.6177 |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |        99 |        0.6869 |        0.6364 |  0.0505 | -0.0505 |  0.1515 |          16 |          11 |   0.6177 |
| CONSENSUS: diverse vs unanimous_plain                               |        99 |        0.8283 |        0.8687 | -0.0404 | -0.1212 |  0.0404 |           7 |          11 |   0.6177 |
| LEXICAL: neutral(varied) vs plain(identical)                        |        99 |        0.899  |        0.8687 |  0.0303 | -0.0404 |  0.101  |           8 |           5 |   0.6177 |
| LEXICAL: confident(varied) vs plain(identical)                      |        99 |        0.899  |        0.8687 |  0.0303 | -0.0404 |  0.1111 |           9 |           6 |   0.6177 |
| ALT OPTION: trust_da vs trust                                       |        99 |        0.6566 |        0.6162 |  0.0404 | -0.0808 |  0.1616 |          20 |          16 |   0.6177 |

### instruct_sft

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| LEXICAL: confident(varied) vs plain(identical)                      |       146 |        0.8425 |        0.6986 |  0.1438 |  0.0685 |  0.2192 |          27 |           6 |   0.0032 |
| LEXICAL: neutral(varied) vs plain(identical)                        |       146 |        0.8356 |        0.6986 |  0.137  |  0.0685 |  0.2055 |          25 |           5 |   0.0032 |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       146 |        0.5274 |        0.6986 | -0.1712 | -0.2603 | -0.0822 |          12 |          37 |   0.0042 |
| CONSENSUS: diverse vs unanimous_plain                               |       146 |        0.5822 |        0.6986 | -0.1164 | -0.1986 | -0.0342 |          11 |          28 |   0.0758 |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       146 |        0.6233 |        0.6986 | -0.0753 | -0.1438 | -0.0137 |           7 |          18 |   0.303  |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       146 |        0.6096 |        0.7123 | -0.1027 | -0.1986 | -0.0068 |          21 |          36 |   0.3764 |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       146 |        0.6096 |        0.5274 |  0.0822 | -0.0137 |  0.1781 |          32 |          20 |   0.6317 |
| REPETITION 1 vs 5 (frame const)                                     |       146 |        0.7466 |        0.6986 |  0.0479 | -0.0342 |  0.1301 |          21 |          14 |   1      |
| TONE: uncertain vs plain                                            |       146 |        0.7397 |        0.6986 |  0.0411 | -0.0411 |  0.1233 |          21 |          15 |   1      |
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       146 |        0.7123 |        0.6986 |  0.0137 | -0.0822 |  0.1096 |          26 |          24 |   1      |
| ALT OPTION: trust_da vs trust                                       |       146 |        0.5205 |        0.5274 | -0.0068 | -0.0822 |  0.0685 |          16 |          17 |   1      |

### instruct_dpo

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       142 |        0.6338 |        0.4437 |  0.1901 |  0.0915 |  0.2887 |          43 |          16 |   0.0064 |
| TONE: uncertain vs plain                                            |       142 |        0.4085 |        0.5352 | -0.1268 | -0.2113 | -0.0423 |          10 |          28 |   0.051  |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       142 |        0.6338 |        0.507  |  0.1268 |  0.0282 |  0.2254 |          36 |          18 |   0.1785 |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       142 |        0.4437 |        0.5352 | -0.0915 | -0.1901 |  0.007  |          21 |          34 |   0.8383 |
| LEXICAL: confident(varied) vs plain(identical)                      |       142 |        0.6056 |        0.5352 |  0.0704 | -0.0211 |  0.162  |          27 |          17 |   1      |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       142 |        0.493  |        0.5352 | -0.0423 | -0.1127 |  0.0282 |          10 |          16 |   1      |
| ALT OPTION: trust_da vs trust                                       |       142 |        0.493  |        0.4437 |  0.0493 | -0.0493 |  0.1408 |          28 |          21 |   1      |
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       142 |        0.507  |        0.5352 | -0.0282 | -0.1268 |  0.0704 |          22 |          26 |   1      |
| CONSENSUS: diverse vs unanimous_plain                               |       142 |        0.5493 |        0.5352 |  0.0141 | -0.0634 |  0.0915 |          17 |          15 |   1      |
| REPETITION 1 vs 5 (frame const)                                     |       142 |        0.5211 |        0.5352 | -0.0141 | -0.1127 |  0.0845 |          26 |          28 |   1      |
| LEXICAL: neutral(varied) vs plain(identical)                        |       142 |        0.5282 |        0.5352 | -0.007  | -0.0915 |  0.0704 |          17 |          18 |   1      |

### instruct

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| LEXICAL: confident(varied) vs plain(identical)                      |       128 |        0.8281 |        0.6172 |  0.2109 |  0.1328 |  0.2891 |          29 |           2 |   0      |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       128 |        0.4453 |        0.6172 | -0.1719 | -0.2812 | -0.0547 |          19 |          41 |   0.0622 |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       128 |        0.5781 |        0.4453 |  0.1328 |  0.0312 |  0.2344 |          32 |          15 |   0.1676 |
| TONE: uncertain vs plain                                            |       128 |        0.5156 |        0.6172 | -0.1016 | -0.1875 | -0.0156 |          10 |          23 |   0.2807 |
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       128 |        0.5156 |        0.6172 | -0.1016 | -0.2031 |  0.0078 |          18 |          31 |   0.598  |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       128 |        0.5781 |        0.5156 |  0.0625 | -0.0312 |  0.1562 |          23 |          15 |   0.8642 |
| ALT OPTION: trust_da vs trust                                       |       128 |        0.5    |        0.4453 |  0.0547 | -0.0391 |  0.1484 |          22 |          15 |   0.8642 |
| CONSENSUS: diverse vs unanimous_plain                               |       128 |        0.6562 |        0.6172 |  0.0391 | -0.0469 |  0.125  |          19 |          14 |   0.8642 |
| REPETITION 1 vs 5 (frame const)                                     |       128 |        0.6562 |        0.6172 |  0.0391 | -0.0547 |  0.1406 |          23 |          18 |   0.8642 |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       128 |        0.6328 |        0.6172 |  0.0156 | -0.0625 |  0.0938 |          14 |          12 |   0.8642 |
| LEXICAL: neutral(varied) vs plain(identical)                        |       128 |        0.6016 |        0.6172 | -0.0156 | -0.1016 |  0.0703 |          16 |          18 |   0.8642 |

### think_sft

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       198 |        0.4242 |        0.9091 | -0.4848 | -0.5707 | -0.399  |          12 |         108 |        0 |
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       198 |        0.4242 |        0.8939 | -0.4697 | -0.5556 | -0.3838 |          12 |         105 |        0 |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       198 |        0.9343 |        0.8939 |  0.0404 | -0.0051 |  0.0859 |          15 |           7 |        1 |
| ALT OPTION: trust_da vs trust                                       |       198 |        0.9444 |        0.9091 |  0.0354 | -0.0051 |  0.0758 |          12 |           5 |        1 |
| LEXICAL: neutral(varied) vs plain(identical)                        |       198 |        0.9293 |        0.8939 |  0.0354 | -0.0051 |  0.0808 |          13 |           6 |        1 |
| LEXICAL: confident(varied) vs plain(identical)                      |       198 |        0.9141 |        0.8939 |  0.0202 | -0.0253 |  0.0657 |          12 |           8 |        1 |
| TONE: uncertain vs plain                                            |       198 |        0.9141 |        0.8939 |  0.0202 | -0.0303 |  0.0707 |          15 |          11 |        1 |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       198 |        0.9091 |        0.8939 |  0.0152 | -0.0303 |  0.0606 |          13 |          10 |        1 |
| CONSENSUS: diverse vs unanimous_plain                               |       198 |        0.9091 |        0.8939 |  0.0152 | -0.0354 |  0.0657 |          14 |          11 |        1 |
| REPETITION 1 vs 5 (frame const)                                     |       198 |        0.9091 |        0.8939 |  0.0152 | -0.0354 |  0.0657 |          15 |          12 |        1 |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       198 |        0.4242 |        0.4242 |  0      | -0.0758 |  0.0758 |          27 |          27 |        1 |

### think_dpo

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       221 |        0.3484 |        0.9367 | -0.5882 | -0.6561 | -0.5158 |           5 |         135 |   0      |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       221 |        0.3303 |        0.9186 | -0.5882 | -0.6606 | -0.5158 |           6 |         136 |   0      |
| ALT OPTION: trust_da vs trust                                       |       221 |        0.9502 |        0.9186 |  0.0317 |  0      |  0.0633 |          10 |           3 |   0.8306 |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       221 |        0.9186 |        0.9367 | -0.0181 | -0.0588 |  0.0181 |           7 |          11 |   1      |
| CONSENSUS: diverse vs unanimous_plain                               |       221 |        0.9186 |        0.9367 | -0.0181 | -0.0588 |  0.0226 |           9 |          13 |   1      |
| REPETITION 1 vs 5 (frame const)                                     |       221 |        0.9231 |        0.9367 | -0.0136 | -0.0543 |  0.0271 |           9 |          12 |   1      |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       221 |        0.3303 |        0.3484 | -0.0181 | -0.0905 |  0.0543 |          32 |          36 |   1      |
| LEXICAL: neutral(varied) vs plain(identical)                        |       221 |        0.9457 |        0.9367 |  0.009  | -0.0181 |  0.0362 |           6 |           4 |   1      |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       221 |        0.9457 |        0.9367 |  0.009  | -0.0271 |  0.0452 |           9 |           7 |   1      |
| LEXICAL: confident(varied) vs plain(identical)                      |       221 |        0.9276 |        0.9367 | -0.009  | -0.0452 |  0.0271 |           7 |           9 |   1      |
| TONE: uncertain vs plain                                            |       221 |        0.9321 |        0.9367 | -0.0045 | -0.0407 |  0.0271 |           7 |           8 |   1      |

### think

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       218 |        0.4037 |        0.9404 | -0.5367 | -0.6101 | -0.4632 |           7 |         124 |   0      |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       218 |        0.4128 |        0.8991 | -0.4862 | -0.5596 | -0.4083 |           9 |         115 |   0      |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       218 |        0.8991 |        0.9404 | -0.0413 | -0.0826 |  0      |           6 |          15 |   0.7052 |
| ALT OPTION: trust_da vs trust                                       |       218 |        0.9266 |        0.8991 |  0.0275 | -0.0092 |  0.0642 |          11 |           5 |   1      |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       218 |        0.9587 |        0.9404 |  0.0183 | -0.0138 |  0.0505 |           8 |           4 |   1      |
| REPETITION 1 vs 5 (frame const)                                     |       218 |        0.9266 |        0.9404 | -0.0138 | -0.0505 |  0.0229 |           7 |          10 |   1      |
| TONE: uncertain vs plain                                            |       218 |        0.9312 |        0.9404 | -0.0092 | -0.0413 |  0.0229 |           6 |           8 |   1      |
| CONSENSUS: diverse vs unanimous_plain                               |       218 |        0.9495 |        0.9404 |  0.0092 | -0.0275 |  0.0459 |          10 |           8 |   1      |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       218 |        0.4128 |        0.4037 |  0.0092 | -0.0734 |  0.0872 |          42 |          40 |   1      |
| LEXICAL: neutral(varied) vs plain(identical)                        |       218 |        0.945  |        0.9404 |  0.0046 | -0.0321 |  0.0413 |           8 |           7 |   1      |
| LEXICAL: confident(varied) vs plain(identical)                      |       218 |        0.9404 |        0.9404 |  0      | -0.0367 |  0.0367 |           9 |           9 |   1      |

## Pre-specified structural contrasts (pooled) — abandonment(a) − abandonment(b), item-bootstrap 95% CI, exact McNemar, Holm within variant

### base

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       636 |        0.6494 |        0.8223 | -0.173  | -0.217  | -0.1305 |          54 |         164 |   0      |
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       636 |        0.6761 |        0.8223 | -0.1462 | -0.1887 | -0.1038 |          58 |         151 |   0      |
| REPETITION 1 vs 5 (frame const)                                     |       636 |        0.717  |        0.8223 | -0.1053 | -0.1462 | -0.0645 |          54 |         121 |   0      |
| CONSENSUS: diverse vs unanimous_plain                               |       636 |        0.7563 |        0.8223 | -0.066  | -0.1053 | -0.0267 |          62 |         104 |   0.0111 |
| TONE: uncertain vs plain                                            |       636 |        0.8695 |        0.8223 |  0.0472 |  0.0126 |  0.0802 |          79 |          49 |   0.0706 |
| LEXICAL: confident(varied) vs plain(identical)                      |       636 |        0.8522 |        0.8223 |  0.0299 | -0.0047 |  0.0629 |          71 |          52 |   0.6252 |
| ALT OPTION: trust_da vs trust                                       |       636 |        0.684  |        0.6494 |  0.0346 | -0.011  |  0.0786 |         118 |          96 |   0.6455 |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       636 |        0.8333 |        0.8223 |  0.011  | -0.0204 |  0.0425 |          57 |          50 |   0.6455 |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       636 |        0.6635 |        0.6494 |  0.0142 | -0.033  |  0.0613 |         121 |         112 |   0.6455 |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       636 |        0.6635 |        0.6761 | -0.0126 | -0.055  |  0.0299 |          92 |         100 |   0.6455 |
| LEXICAL: neutral(varied) vs plain(identical)                        |       636 |        0.8318 |        0.8223 |  0.0094 | -0.0252 |  0.0425 |          62 |          56 |   0.6455 |

### instruct_sft

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| LEXICAL: neutral(varied) vs plain(identical)                        |       834 |        0.8609 |        0.717  |  0.1439 |  0.1115 |  0.1763 |         163 |          43 |   0      |
| LEXICAL: confident(varied) vs plain(identical)                      |       834 |        0.8609 |        0.717  |  0.1439 |  0.1115 |  0.1763 |         165 |          45 |   0      |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       834 |        0.5576 |        0.717  | -0.1595 | -0.1954 | -0.1223 |          63 |         196 |   0      |
| CONSENSUS: diverse vs unanimous_plain                               |       834 |        0.5995 |        0.717  | -0.1175 | -0.1523 | -0.0815 |          74 |         172 |   0      |
| TONE: uncertain vs plain                                            |       834 |        0.8141 |        0.717  |  0.0971 |  0.0635 |  0.1307 |         146 |          65 |   0      |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       834 |        0.6295 |        0.7146 | -0.0851 | -0.1271 | -0.0432 |         119 |         190 |   0.0004 |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       834 |        0.6295 |        0.5576 |  0.0719 |  0.0324 |  0.1115 |         170 |         110 |   0.002  |
| REPETITION 1 vs 5 (frame const)                                     |       834 |        0.7374 |        0.717  |  0.0204 | -0.0144 |  0.0552 |         120 |         103 |   0.8721 |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       834 |        0.7014 |        0.717  | -0.0156 | -0.0432 |  0.0108 |          58 |          71 |   0.8721 |
| ALT OPTION: trust_da vs trust                                       |       834 |        0.5492 |        0.5576 | -0.0084 | -0.0444 |  0.0276 |         113 |         120 |   0.9504 |
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       834 |        0.7146 |        0.717  | -0.0024 | -0.0408 |  0.0348 |         128 |         130 |   0.9504 |

### instruct_dpo

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       862 |        0.6148 |        0.4501 |  0.1647 |  0.1241 |  0.2053 |         243 |         101 |   0      |
| TONE: uncertain vs plain                                            |       862 |        0.4501 |        0.5847 | -0.1346 | -0.1705 | -0.0986 |          71 |         187 |   0      |
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       862 |        0.4501 |        0.5847 | -0.1346 | -0.1752 | -0.0928 |         113 |         229 |   0      |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       862 |        0.6148 |        0.5035 |  0.1114 |  0.0719 |  0.1508 |         201 |         105 |   0      |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       862 |        0.5035 |        0.5847 | -0.0812 | -0.123  | -0.0394 |         138 |         208 |   0.0014 |
| LEXICAL: confident(varied) vs plain(identical)                      |       862 |        0.6555 |        0.5847 |  0.0708 |  0.0336 |  0.1079 |         165 |         104 |   0.0014 |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       862 |        0.5742 |        0.5847 | -0.0104 | -0.0429 |  0.0209 |          91 |         100 |   1      |
| ALT OPTION: trust_da vs trust                                       |       862 |        0.5151 |        0.5035 |  0.0116 | -0.0255 |  0.0487 |         137 |         127 |   1      |
| LEXICAL: neutral(varied) vs plain(identical)                        |       862 |        0.5789 |        0.5847 | -0.0058 | -0.0429 |  0.0313 |         130 |         135 |   1      |
| REPETITION 1 vs 5 (frame const)                                     |       862 |        0.5905 |        0.5847 |  0.0058 | -0.0313 |  0.0441 |         140 |         135 |   1      |
| CONSENSUS: diverse vs unanimous_plain                               |       862 |        0.5847 |        0.5847 |  0      | -0.0336 |  0.0348 |         120 |         120 |   1      |

### instruct

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |       802 |        0.4663 |        0.6721 | -0.2057 | -0.2494 | -0.1621 |          93 |         258 |   0      |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |       802 |        0.4875 |        0.6721 | -0.1845 | -0.2269 | -0.1421 |          94 |         242 |   0      |
| LEXICAL: confident(varied) vs plain(identical)                      |       802 |        0.8217 |        0.6721 |  0.1496 |  0.1147 |  0.1858 |         175 |          55 |   0      |
| TONE: uncertain vs plain                                            |       802 |        0.5524 |        0.6721 | -0.1197 | -0.1559 | -0.0823 |          69 |         165 |   0      |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |       802 |        0.5948 |        0.4663 |  0.1284 |  0.0885 |  0.1683 |         193 |          90 |   0      |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |       802 |        0.5948 |        0.4875 |  0.1072 |  0.0661 |  0.1484 |         184 |          98 |   0      |
| REPETITION 1 vs 5 (frame const)                                     |       802 |        0.7269 |        0.6721 |  0.0549 |  0.015  |  0.096  |         159 |         115 |   0.0463 |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |       802 |        0.7007 |        0.6721 |  0.0287 | -0.0012 |  0.0586 |          88 |          65 |   0.2998 |
| ALT OPTION: trust_da vs trust                                       |       802 |        0.5162 |        0.4875 |  0.0287 | -0.0075 |  0.0648 |         122 |          99 |   0.4162 |
| CONSENSUS: diverse vs unanimous_plain                               |       802 |        0.6845 |        0.6721 |  0.0125 | -0.0224 |  0.0474 |         104 |          94 |   1      |
| LEXICAL: neutral(varied) vs plain(identical)                        |       802 |        0.6708 |        0.6721 | -0.0012 | -0.0387 |  0.0374 |         116 |         117 |   1      |

### think_sft

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |      1255 |        0.4311 |        0.9171 | -0.4861 | -0.5179 | -0.4542 |          44 |         654 |   0      |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |      1255 |        0.4558 |        0.9116 | -0.4558 | -0.4884 | -0.4239 |          59 |         631 |   0      |
| LEXICAL: confident(varied) vs plain(identical)                      |      1255 |        0.9315 |        0.9171 |  0.0143 | -0.0016 |  0.0303 |          60 |          42 |   0.8265 |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |      1255 |        0.4558 |        0.4311 |  0.0247 | -0.0064 |  0.0542 |         208 |         177 |   1      |
| ALT OPTION: trust_da vs trust                                       |      1255 |        0.9235 |        0.9116 |  0.012  | -0.004  |  0.0279 |          60 |          45 |   1      |
| CONSENSUS: diverse vs unanimous_plain                               |      1255 |        0.9259 |        0.9171 |  0.0088 | -0.008  |  0.0263 |          64 |          53 |   1      |
| LEXICAL: neutral(varied) vs plain(identical)                        |      1255 |        0.9251 |        0.9171 |  0.008  | -0.0072 |  0.0239 |          57 |          47 |   1      |
| REPETITION 1 vs 5 (frame const)                                     |      1255 |        0.9084 |        0.9171 | -0.0088 | -0.0263 |  0.0096 |          59 |          70 |   1      |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |      1255 |        0.9243 |        0.9171 |  0.0072 | -0.0096 |  0.0239 |          62 |          53 |   1      |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |      1255 |        0.9116 |        0.9171 | -0.0056 | -0.0215 |  0.0112 |          53 |          60 |   1      |
| TONE: uncertain vs plain                                            |      1255 |        0.9179 |        0.9171 |  0.0008 | -0.0167 |  0.0175 |          60 |          59 |   1      |

### think_dpo

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |      1251 |        0.4053 |        0.8209 | -0.4157 | -0.4492 | -0.3813 |          93 |         613 |   0      |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |      1251 |        0.4141 |        0.7842 | -0.3701 | -0.4061 | -0.3333 |         126 |         589 |   0      |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |      1251 |        0.7842 |        0.8209 | -0.0368 | -0.056  | -0.0176 |          50 |          96 |   0.0016 |
| REPETITION 1 vs 5 (frame const)                                     |      1251 |        0.7834 |        0.8209 | -0.0376 | -0.0576 | -0.0176 |          56 |         103 |   0.0019 |
| ALT OPTION: trust_da vs trust                                       |      1251 |        0.8058 |        0.7842 |  0.0216 |  0.0024 |  0.0416 |          93 |          66 |   0.2721 |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |      1256 |        0.4124 |        0.4045 |  0.008  | -0.0231 |  0.0398 |         209 |         199 |   1      |
| LEXICAL: confident(varied) vs plain(identical)                      |      1251 |        0.8161 |        0.8209 | -0.0048 | -0.0224 |  0.0136 |          66 |          72 |   1      |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |      1251 |        0.8249 |        0.8209 |  0.004  | -0.0152 |  0.0232 |          74 |          69 |   1      |
| CONSENSUS: diverse vs unanimous_plain                               |      1251 |        0.8193 |        0.8209 | -0.0016 | -0.0224 |  0.0192 |          86 |          88 |   1      |
| LEXICAL: neutral(varied) vs plain(identical)                        |      1251 |        0.8209 |        0.8209 |  0      | -0.0184 |  0.0184 |          69 |          69 |   1      |
| TONE: uncertain vs plain                                            |      1251 |        0.8217 |        0.8209 |  0.0008 | -0.0192 |  0.0208 |          80 |          79 |   1      |

### think

| contrast                                                            |   n_pairs |   p_abandon_a |   p_abandon_b |    diff |      lo |      hi |   mcnemar_b |   mcnemar_c |   p_holm |
|:--------------------------------------------------------------------|----------:|--------------:|--------------:|--------:|--------:|--------:|------------:|------------:|---------:|
| FRAME+SYS: prior-users/warn vs participant/ctrl (5 ident both)      |      1338 |        0.4387 |        0.9335 | -0.4948 | -0.5239 | -0.4649 |          28 |         690 |   0      |
| SYS PROMPT+wording: auth_bias(warn) vs auth_trust(ctrl)             |      1338 |        0.4275 |        0.9155 | -0.488  | -0.5187 | -0.4574 |          50 |         703 |   0      |
| CONSENSUS: DA(4+1) vs unanimous_plain                               |      1338 |        0.9559 |        0.9335 |  0.0224 |  0.0097 |  0.0351 |          52 |          22 |   0.0058 |
| SINGLE CLAIM vs 5 REPEATS (ctrl sys): auth_trust vs unanimous_plain |      1338 |        0.9155 |        0.9335 | -0.0179 | -0.0336 | -0.0022 |          47 |          71 |   0.2703 |
| CONSENSUS: diverse vs unanimous_plain                               |      1338 |        0.9432 |        0.9335 |  0.0097 | -0.0052 |  0.0247 |          57 |          44 |   1      |
| SINGLE CLAIM vs 5 REPEATS (warn sys): auth_bias vs asch_history_5   |      1338 |        0.4275 |        0.4387 | -0.0112 | -0.0419 |  0.0202 |         213 |         228 |   1      |
| REPETITION 1 vs 5 (frame const)                                     |      1338 |        0.9283 |        0.9335 | -0.0052 | -0.0194 |  0.009  |          44 |          51 |   1      |
| ALT OPTION: trust_da vs trust                                       |      1338 |        0.9208 |        0.9155 |  0.0052 | -0.0105 |  0.0209 |          59 |          52 |   1      |
| LEXICAL: confident(varied) vs plain(identical)                      |      1338 |        0.9372 |        0.9335 |  0.0037 | -0.0097 |  0.0172 |          47 |          42 |   1      |
| TONE: uncertain vs plain                                            |      1338 |        0.935  |        0.9335 |  0.0015 | -0.012  |  0.0149 |          45 |          43 |   1      |
| LEXICAL: neutral(varied) vs plain(identical)                        |      1338 |        0.9327 |        0.9335 | -0.0007 | -0.0142 |  0.0127 |          40 |          41 |   1      |

## Condition-ordering stability across temperatures (Kendall's W over 6 temps × 11 conditions)

| variant      |   kendall_W |   n_temps |   n_conditions | mean_order                                                                                                                                                                                                                                  |
|:-------------|------------:|----------:|---------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| base         |       0.777 |         6 |             11 | unan_uncertain(5 var) > unan_confident(5 var) > da(4+1) > unan_neutral(5 var) > unan_plain(5 ident) > diverse(no consensus) > qd(1 stated) > auth_trust_da(1+alt) > asch_hist(5 ident,warn,prior-users) > auth_bias(1,warn) > auth_trust(1) |
| instruct_sft |       0.951 |         6 |             11 | unan_confident(5 var) > unan_neutral(5 var) > unan_uncertain(5 var) > qd(1 stated) > unan_plain(5 ident) > asch_hist(5 ident,warn,prior-users) > da(4+1) > auth_bias(1,warn) > diverse(no consensus) > auth_trust(1) > auth_trust_da(1+alt) |
| instruct_dpo |       0.81  |         6 |             11 | unan_confident(5 var) > auth_bias(1,warn) > qd(1 stated) > unan_plain(5 ident) > diverse(no consensus) > unan_neutral(5 var) > da(4+1) > auth_trust_da(1+alt) > auth_trust(1) > asch_hist(5 ident,warn,prior-users) > unan_uncertain(5 var) |
| instruct     |       0.925 |         6 |             11 | unan_confident(5 var) > qd(1 stated) > da(4+1) > diverse(no consensus) > unan_plain(5 ident) > unan_neutral(5 var) > auth_bias(1,warn) > unan_uncertain(5 var) > auth_trust_da(1+alt) > auth_trust(1) > asch_hist(5 ident,warn,prior-users) |
| think_sft    |       0.592 |         6 |             11 | unan_confident(5 var) > diverse(no consensus) > unan_neutral(5 var) > da(4+1) > auth_trust_da(1+alt) > unan_uncertain(5 var) > unan_plain(5 ident) > auth_trust(1) > qd(1 stated) > auth_bias(1,warn) > asch_hist(5 ident,warn,prior-users) |
| think_dpo    |       0.562 |         6 |             11 | da(4+1) > unan_uncertain(5 var) > unan_plain(5 ident) > unan_neutral(5 var) > diverse(no consensus) > unan_confident(5 var) > auth_trust_da(1+alt) > auth_trust(1) > qd(1 stated) > auth_bias(1,warn) > asch_hist(5 ident,warn,prior-users) |
| think        |       0.713 |         6 |             11 | da(4+1) > diverse(no consensus) > unan_confident(5 var) > unan_uncertain(5 var) > unan_plain(5 ident) > unan_neutral(5 var) > qd(1 stated) > auth_trust_da(1+alt) > auth_trust(1) > asch_hist(5 ident,warn,prior-users) > auth_bias(1,warn) |
