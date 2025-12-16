This folder contains machine-readable "immutable facts" items used by the Olmo conformity suite.

Format: JSONL (one JSON object per line).
Required keys per item:
  - item_id: stable string id
  - domain: e.g. "geography", "math"
  - question: question string
  - ground_truth_text: canonical answer string
  - source: object with provenance (dataset name, split, etc.)

Do not put large benchmark dumps here. Keep small curated subsets or pointers.


