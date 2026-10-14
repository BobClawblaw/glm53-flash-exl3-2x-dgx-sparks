| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| gsm8k | 244/250 | 97.6% | 196s | misses at [12, 62, 87, 119, 172, 173] |
| mmlu | 292/342 | 85.4% | 437s | 57 subjects x 6; weakest virology 2/6, business_ethics 3/6, college_chemistry 3/6 |
| humaneval | 156/164 | 95.1% | 187s | pass@1 greedy; failed ['HumanEval_130', 'HumanEval_132', 'HumanEval_145', 'HumanEval_158', 'HumanEval_163', 'HumanEval_50'] ... |
| ifeval | 118/150 | 78.7% | 564s | loose 118/150, strict 111/150, 150 covered prompts |
| tools | 59/60 | 98.3% | 84s | 'Drive me from Austin to Dallas.': got [('get_directions', {'destination': 'dallas', 'origin': 'austin'})] |
| json | 30/30 | 100.0% | 13s | all valid |
| mgsm | 223/240 | 92.9% | 329s | de 28/30, es 29/30, fr 27/30, ja 26/30, zh 30/30, ru 29/30, sw 26/30, bn 28/30 |
