---
dataset_info:
  features:
  - name: ground_truth
    list: string
  - name: dataset
    list: string
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
  - name: dataset_source
    dtype: string
  - name: input_ids_prompt
    list: int64
  - name: input_ids
    list: int32
  - name: attention_mask
    list: int8
  - name: labels
    list: int64
  - name: prompt
    dtype: string
  - name: id
    dtype: string
  - name: key
    dtype: string
  - name: constraint_type
    dtype: string
  - name: constraint
    dtype: string
  - name: conversation_hash
    dtype: string
  - name: model
    dtype: string
  - name: predicted_label
    dtype: string
  splits:
  - name: train
    num_bytes: 4083445248
    num_examples: 102014
  download_size: 1893783057
  dataset_size: 4083445248
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Dolci-Think-RL-7B

## Dataset Summary

**Dolci-Think-RL-7B** is the reinforcement learning dataset used to train the *Olmo-3-7B-Think* model.  
It contains **102,014** prompts designed to elicit deep reasoning across:

- Math
- Coding
- Precise Instruction Following
- General Chat

It blends high-quality curated sources with filtering designed for deliberate reasoning.

---

## Dataset Composition

### **Total Samples:** 102,014

### **Original Dataset Contribution**
| Source Dataset | Count |
|----------------|-------|
| IF Multi-Constraint | 29,813 |
| OMEGA Math ([paper](https://arxiv.org/abs/2506.18880)) | 15,000 |
| AceCoder ([paper](https://arxiv.org/abs/2502.01718)) | 10,107 |
| Tulu 3 Rewritten ([paper](https://arxiv.org/abs/2411.15124)) | 7,109 |
| Multi-Subject RLVR ([paper](https://arxiv.org/abs/2503.23829v1)) | 7,106 |
| AceReason-Math ([paper](https://arxiv.org/abs/2505.16400)) | 6,598 |
| WildChat English ([paper](https://arxiv.org/abs/2405.01470)) | 6,421 |
| KlearReasoner Code | 6,272 |
| SYNTHETIC-2 / PrimeIntellect ([blog](https://www.primeintellect.ai/blog/synthetic-2)) | 3,000 |
| MathSub-30K (KlearReasoner Math) ([paper](https://arxiv.org/abs/2508.07629)) | 2,999 |
| ORZ Math ([paper](https://arxiv.org/abs/2503.24290)) | 2,999 |
| DAPO-Math ([paper](https://arxiv.org/abs/2503.14476)) | 2,584 |
| Llama-Nemotron Post-Training Dataset ([paper](https://arxiv.org/abs/2505.00949)) | 2,006 |

### **Dataset Source Counts (Grouped Mixes)**
| Mix | Count |
|------|-------|
| Math RLVR Mixture | 30,180 |
| IF RLVR Mixture | 29,813 |
| Code RLVR Mixture | 21,385 |
| General RLVR Mixture | 20,636 |

---

## Data Sources & Description

### **Instruction Following**
- Up to 5 constraints  
- Derived from IFBench-Train & IFEval-style tasks  
- Filtered for clarity and non-toxicity  

### **Math Reasoning**
- **OMEGA**  
- **AceReason-Math**  
- **ORZ Math**  
- **DAPO-Math**  
- **MathSub-30K**  
- Wide domain coverage: geometry, algebra, combinatorics, proofs, etc.

### **Code Reasoning**
Includes four major families:
- **AceCoder**  
- **KlearReasoner-Code**  
- **SYNTHETIC-2 / PrimeIntellect**  
- **Llama-Nemotron Post-Training Dataset**  
All filtered via test-case execution.

### **General Long-Form Reasoning**
- Multi-Subject RLVR  
- Tulu 3 rewritten (filtered via F1-score)  
- WildChat English (filtered for reasoning suitability)  

---

## Processing & Filtering

- **Execution-based code filtering** (test-case validated)  
- **Topic filtering** for safety and quality  
- **F1-based rewrite filtering** (Tulu 3)  
- **Difficulty-tiered Nemotron subsets**  
- **Strict deduplication**  
- **Constraint normalization**  

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
