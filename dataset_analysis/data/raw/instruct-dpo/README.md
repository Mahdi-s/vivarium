---
dataset_info:
  features:
  - name: chosen
    list:
    - name: annotations
      dtype: 'null'
    - name: audio
      dtype: 'null'
    - name: content
      dtype: string
    - name: country
      dtype: string
    - name: created
      dtype: 'null'
    - name: function_call
      dtype: 'null'
    - name: hashed_ip
      dtype: string
    - name: header
      struct:
      - name: accept-language
        dtype: string
      - name: user-agent
        dtype: string
    - name: language
      dtype: string
    - name: openai_id
      dtype: 'null'
    - name: reasoning_content
      dtype: 'null'
    - name: redacted
      dtype: bool
    - name: refusal
      dtype: 'null'
    - name: role
      dtype: string
    - name: state
      dtype: string
    - name: system_fingerprint
      dtype: 'null'
    - name: temperature
      dtype: 'null'
    - name: timestamp
      dtype: 'null'
    - name: token_counter
      dtype: 'null'
    - name: tool_calls
      list: 'null'
    - name: top_p
      dtype: 'null'
    - name: toxic
      dtype: bool
    - name: turn_identifier
      dtype: int64
    - name: usage
      dtype: 'null'
  - name: rejected
    list:
    - name: annotations
      dtype: 'null'
    - name: audio
      dtype: 'null'
    - name: content
      dtype: string
    - name: country
      dtype: string
    - name: created
      dtype: 'null'
    - name: function_call
      dtype: 'null'
    - name: hashed_ip
      dtype: string
    - name: header
      struct:
      - name: accept-language
        dtype: string
      - name: user-agent
        dtype: string
    - name: language
      dtype: string
    - name: openai_id
      dtype: 'null'
    - name: reasoning_content
      dtype: 'null'
    - name: redacted
      dtype: bool
    - name: refusal
      dtype: 'null'
    - name: role
      dtype: string
    - name: state
      dtype: string
    - name: system_fingerprint
      dtype: 'null'
    - name: temperature
      dtype: 'null'
    - name: timestamp
      dtype: 'null'
    - name: token_counter
      dtype: 'null'
    - name: tool_calls
      list: 'null'
    - name: top_p
      dtype: 'null'
    - name: toxic
      dtype: bool
    - name: turn_identifier
      dtype: int64
    - name: usage
      dtype: 'null'
  - name: chosen_model
    dtype: string
  - name: rejected_model
    dtype: string
  - name: prompt_id
    dtype: string
  - name: preference_type
    dtype: string
  splits:
  - name: train
    num_bytes: 1783933715
    num_examples: 259922
  download_size: 810233679
  dataset_size: 1783933715
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: odc-by
---

# Dolci Instruct DPO Mixture

This dataset is licensed under ODC-BY. It is intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use).

The Dolci Instruct DPO mixture was used to preference tune Olmo 3 Instruct 7B. It contains 260,000 preference pairs in total, including:
- 125,000 pairs created with the preference heuristic described in [Delta Learning](https://arxiv.org/abs/2507.06187) (Geng et al. 2025)
- 125,000 pairs created with a delta-aware Ultrafeedback-esque GPT-judge pipeline, designed to maximize the contrast between chosen and rejected responses
- 10,000 multiturn preference pairs (5,000 synthetic context, 5,000 self talk)

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
