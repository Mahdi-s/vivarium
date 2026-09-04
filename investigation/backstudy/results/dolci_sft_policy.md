# Dolci-Instruct-SFT (1/7 stratified sample, n=307,455): hedging, 'say you are unsure' instructions, definiteness, push-back — by source

Overall: hedge 0.0007 · unsure-instruction in prompt 0.0005 · definite-answer marker 0.1232 · push-back 0.0021

P(hedge | unsure instruction present) = 0.000 vs P(hedge | absent) = 0.001 (n=165 vs 307290)

| source                                |     n |   hedge |   unsure_instr |   definite |   pushback |   has_system |   len_med |
|:--------------------------------------|------:|--------:|---------------:|-----------:|-----------:|-------------:|----------:|
| Verifiable Reasoning                  | 44369 |  0      |         0      |     0.0077 |     0.0007 |       0      |     422   |
| Wildchat                              | 43202 |  0.0027 |         0.0034 |     0.0199 |     0.0039 |       0      |    2092   |
| Dolci Instruct Tool Use               | 32512 |  0.0002 |         0      |     0.0453 |     0.0007 |       1      |     487   |
| Dolci Instruct Python Algorithms      | 26910 |  0.0001 |         0      |     0.0014 |     0.0026 |       0      |     678   |
| Logic Puzzles                         | 22971 |  0.0016 |         0      |     0.0401 |     0.0001 |       0      |      19   |
| Tulu 3 Persona MATH                   | 21416 |  0.0001 |         0      |     0.9967 |     0.0013 |       0      |    2673.5 |
| Dolci Instruct Precise IF             | 19490 |  0.0003 |         0.0001 |     0.0802 |     0.0007 |       0      |     858   |
| Evol CodeAlpaca                       | 15384 |  0.0008 |         0      |     0.0032 |     0.0069 |       0      |    1460   |
| Dolci Instruct OpenThoughts3+ Science | 14174 |  0      |         0      |     0.2756 |     0.0012 |       0      |    2304   |
| Aya                                   | 14042 |  0.0003 |         0      |     0.0001 |     0      |       0      |     125   |
| FLAN                                  | 12793 |  0.0001 |         0.001  |     0.297  |     0.0003 |       0      |      80   |
| OpenMathInstruct 2                    |  7193 |  0      |         0      |     0.0452 |     0.0006 |       0      |     463   |
| WildGuardMix                          |  7056 |  0.0016 |         0.0003 |     0.0004 |     0.0136 |       0      |     435   |
| WildJailbreak                         |  7040 |  0.0011 |         0      |     0.0001 |     0.0091 |       0      |    1319   |
| Tulu 3 Persona GSM                    |  7035 |  0      |         0      |     0.094  |     0      |       0      |    1142   |
| Tulu 3 Persona Python                 |  5011 |  0      |         0      |     0      |     0.0008 |       0      |     345   |
| Tulu 3 Persona Algebra                |  2925 |  0      |         0      |     0.8349 |     0      |       0      |    2254   |
| CoCoNot                               |  1565 |  0.0006 |         0      |     0.0064 |     0.0013 |       0      |     975   |
| OpenAssistant                         |  1010 |  0.0059 |         0      |     0      |     0.002  |       0      |    1006   |
| TableGPT                              |   677 |  0      |         0      |     0.2112 |     0      |       0      |     152   |
| SciRiff                               |   669 |  0      |         0      |     0.003  |     0.003  |       0      |     212   |
| Hardcoded Data                        |    11 |  0      |         0      |     0      |     0      |       0.8182 |     321   |
