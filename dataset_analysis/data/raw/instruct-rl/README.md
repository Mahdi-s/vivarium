---
dataset_info:
  features:
  - name: ground_truth
    list: string
  - name: dataset
    list: string
  - name: solution
    dtype: string
  - name: id
    dtype: string
  - name: difficulty
    dtype: int64
  - name: difficulty_explanation
    dtype: string
  - name: dataset_source
    dtype: string
  - name: input_ids_prompt
    list: int64
  - name: prompt
    dtype: string
  - name: setting_key
    dtype: string
  - name: setting_name
    dtype: string
  - name: data_source
    dtype: string
  - name: source_prompt
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: ability
    dtype: string
  - name: reward_model
    struct:
    - name: ground_truth
      dtype: string
    - name: style
      dtype: string
  - name: extra_info
    struct:
    - name: index
      dtype: string
  - name: key
    dtype: string
  - name: constraint_type
    dtype: string
  - name: constraint
    dtype: string
  - name: custom_id
    dtype: string
  - name: original_dataset
    dtype: string
  - name: outputs
    list: string
  - name: total_rollouts
    dtype: int64
  - name: total_correct_rollouts
    dtype: float64
  - name: passrate
    dtype: float64
  - name: topic
    dtype: string
  - name: characters
    list: string
  - name: conversation_hash
    dtype: string
  - name: model
    dtype: string
  - name: predicted_label
    dtype: string
  splits:
  - name: train
    num_bytes: 1127596768
    num_examples: 169964
  download_size: 482557259
  dataset_size: 1127596768
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Dolci-Instruct-RL

## Dataset Summary

**Dolci-Instruct-RL** is the reinforcement learning dataset used to train the *Olmo-3-7B-Instruct* model.  
It contains **169,964** prompts spanning:

- Math
- Code
- Precise Instruction Following
- General Chat

The dataset aggregates multiple curated sources, applies extensive filtering, and produces a unified RL-ready prompt set.

---

## Dataset Composition

### **Total Samples:** 169,964

### **Original Dataset Contribution**
| Source Dataset | Count |
|----------------|-------|
| IF Multi-Constraint (IFBench/IFEval derived) | 37,568 |
| Multi-Subject RLVR ([paper](https://arxiv.org/abs/2503.23829v1)) | 18,971 |
| Tulu 3 Rewritten ([paper](https://arxiv.org/abs/2411.15124)) | 18,757 |
| WildChat English General ([paper](https://arxiv.org/abs/2405.01470)) | 10,670 |

### **Dataset Source Counts (Grouped Mixes)**
| Mix | Count |
|------|-------|
| General RLVR Mix | 48,398 |
| IF Multi-Constraint Mixture | 37,568 |
| AceCoder RLVR ([paper](https://arxiv.org/abs/2502.01718)) | 20,000 |
| OMEGA (Math) ([paper](https://arxiv.org/abs/2506.18880)) | 20,000 |
| ORZ Math (Open-Reasoner-Zero) ([paper](https://arxiv.org/abs/2503.24290)) | 14,000 |
| Polaris Math | 14,000 |
| MathSub-30K (KlearReasoner Math) ([paper](https://arxiv.org/abs/2508.07629)) | 8,998 |
| DAPO-Math ([paper](https://arxiv.org/abs/2503.14476)) | 7,000 |

---

## Data Sources & Description

### **Instruction Following**
- Derived from IFBench-Train & IFEval-style prompts  
- Strict multi-constraint format (up to 5 constraints)
- Normalized and filtered for safety and clarity

### **General Chat**
- **Tulu 3 Rewritten** prompts (clarified and F1 filtered)  
- **WildChat English** (filtered for non-math, non-code; character caps)  
- **Multi-Subject RLVR** exam-style reasoning questions  

### **Math**
- **OMEGA** ([paper](https://arxiv.org/abs/2506.18880))  
- **Open-Reasoner-Zero (ORZ)** ([paper](https://arxiv.org/abs/2503.24290))  
- **DAPO-Math** ([paper](https://arxiv.org/abs/2503.14476))  
- **MathSub-30K (KlearReasoner Math)** ([paper](https://arxiv.org/abs/2508.07629))  
- **Polaris**  

### **Code**
- **AceCoder** ([paper](https://arxiv.org/abs/2502.01718))  
- Test-case–based RL prompts  
- High-quality filtering via solution execution  
- Some test cases synthesized programmatically  

---

## Processing & Filtering

- **Keyword & topic filtering**  
- **Character caps** (max 10 per character for WildChat)  
- **F1-quality screening** for Tulu 3 rewritten prompts  
- **Removal of math/code** from general-chat datasets  
- **Execution-based filtering** for code datasets  
- **Constraint normalization** for IF prompts  

The final result is a clean, high-entropy, instruction-following RL dataset.

---

## License
This dataset is licensed under ODC-BY. It is intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).

## Citation

```
@misc{olmo2025olmo3,
title={Olmo 3},
author={Team Olmo and Allyson Ettinger and Amanda Bertsch and Bailey Kuehl and David Graham and David Heineman and Dirk Groeneveld and Faeze Brahman and Finbarr Timbers and Hamish Ivison and Jacob Morrison and Jake Poznanski and Kyle Lo and Luca Soldaini and Matt Jordan and Mayee Chen and Michael Noukhovitch and Nathan Lambert and Pete Walsh and Pradeep Dasigi and Robert Berry and Saumya Malik and Saurabh Shah and Scott Geng and Shane Arora and Shashank Gupta and Taira Anderson and Teng Xiao and Tyler Murray and Tyler Romero and Victoria Graf and Akari Asai and Akshita Bhagia and Alexander Wettig and Alisa Liu and Aman Rangapur and Chloe Anastasiades and Costa Huang and Dustin Schwenk and Harsh Trivedi and Ian Magnusson and Jaron Lochner and Jiacheng Liu and Lester James V. Miranda and Maarten Sap and Malia Morgan and Michael Schmitz and Michal Guerquin and Michael Wilson and Regan Huff and Ronan Le Bras and Rui Xin and Rulin Shao and Sam Skjonsberg and Shannon Zejiang Shen and Shuyue Stella Li and Tucker Wilde and Valentina Pyatkin and Will Merrill and Yapei Chang and Yuling Gu and Zhiyuan Zeng and Ashish Sabharwal and Luke Zettlemoyer and Pang Wei Koh and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
year={2025},
eprint={2512.13961},
archivePrefix={arXiv},
primaryClass={cs.CL},
url={https://arxiv.org/abs/2512.13961},
}
```
