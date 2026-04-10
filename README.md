# Extenics-GNN-Differential-Framework
Official core implementation of the Extension Graph Difference Framework (EGDF) presented in the associated manuscript. 
This repository provides the GNN-based conflict analysis engine and annotated training datasets.
## Repository Structure
EGDF/
├── train.py          # GNN training script with 5-fold Cross-Validation logic
├── inference.py      # GNN inference module for generating structural guiding signals
├── data/             # Annotated training dataset (200 total samples)
│   ├── ecological/  
│   ├── financial/    
│   ├── medical/      
│   └── production/   
└── requirements.txt  
