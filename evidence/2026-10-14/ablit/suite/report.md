| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| gsm8k | 246/250 | 98.4% | 212s | misses at [12, 89, 119, 172] |
| mmlu | 292/342 | 85.4% | 442s | 57 subjects x 6; weakest high_school_government_and_politics 2/6, business_ethics 3/6, college_chemistry 3/6 |
| humaneval | 159/164 | 97.0% | 198s | pass@1 greedy; failed ['HumanEval_108', 'HumanEval_132', 'HumanEval_145', 'HumanEval_163', 'HumanEval_93'] |
| ifeval | 118/150 | 78.7% | 516s | loose 118/150, strict 109/150, 150 covered prompts |
| tools | 59/60 | 98.3% | 83s | 'Drive me from Austin to Dallas.': got [('get_directions', {'destination': 'dallas', 'origin': 'austin'})] |
| json | 30/30 | 100.0% | 14s | all valid |
| mgsm | 221/240 | 92.1% | 290s | de 26/30, es 28/30, fr 28/30, ja 27/30, zh 28/30, ru 29/30, sw 27/30, bn 28/30 |
