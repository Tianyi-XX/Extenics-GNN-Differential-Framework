# Extenics-GNN-Differential-Framework
Official core implementation of the Extension Graph Difference Framework (EGDF) presented in the associated manuscript. 
This repository provides the GNN-based conflict analysis engine and annotated training datasets.
## Repository Structure
```text
EGDF/
├── train.py          # GNN training script with 5-fold Cross-Validation logic
├── inference.py      # GNN inference module for generating structural guiding signals
├── data/             # Expert-annotated training dataset (200 total samples)
│   ├── ecological/   # 50 samples: invasive species & biodiversity conflicts
│   ├── financial/    # 50 samples: risk control & regulatory constraints
│   ├── medical/      # 50 samples: clinical safety & resource allocation
│   └── production/   # 50 samples: capacity, quality, and cost trade-offs
└── requirements.txt  # Environment dependencies
