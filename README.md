# Extenics-GNN-Differential-Framework
Official core implementation of the Extension Graph Difference Framework (EGDF) presented in the associated manuscript. 
This repository provides the GNN-based conflict analysis engine and annotated training datasets.

## Repository Structure
```text
EGDF/
├── train.py          # GNN training script with 5-fold Cross-Validation logic
├── inference.py      # GNN inference module for generating structural guiding signals
├── data/             # Annotated training dataset (200 total samples)
│   ├── ecological/   
│   ├── financial/    
│   ├── medical/      
│   └── production/   
└── requirements.txt
```

## Requirements
```bash
pip install torch torch-geometric scikit-learn numpy
```

## Training & Reproducibility
This module implements the **GNN-based structural analysis** described in Section 3.2 of the paper.

```bash
python train.py
```

- **Multi-Task Optimization**: Implements the composite loss function from Equation (15), coordinating node importance, relevance, edge conflict probability, severity, and problem identification.
- **Statistical Robustness**: Includes 5-fold cross-validation to ensure model performance is insensitive to specific data splits across the four domains.
- **Outputs**: Generates `best_model_structural.pt` (trained weights) and `cv_results_structural.json` (MAE and validation loss metrics).

## Inference
The inference module loads the trained multi-task GNN model to perform quantitative structural analysis on new basic-element graphs.

```bash
python inference.py
```

It automates the Extenics analytical process by outputting:
- **Ranked Key Nodes**: Identifies nodes with high structural centrality based on importance and relevance scores.
- **Quantified Conflicts**: Detects conflict edges with predicted probability and severity scores.
- **Structured Guiding Signals**: Encodes the analysis results into a formatted prompt (as detailed in Appendix A of the paper) to steer the LLM toward targeted transformation strategies.

## Dataset Highlights
The dataset consists of **200 problem-description graphs** modeled using Extenics basic-element theory, with 50 samples drawn from each of the four domains.
- **Formal Modeling**: Each sample follows the matter-element, affair-element, and relation-element ordered triple format (O, c, v).
- **Expert Annotation**: Ground-truth labels were established through the consensus of three doctoral researchers specializing in Extenics theory to ensure high-quality supervisory signals.
- **Graph Structure**: Data is stored in structured JSON format, mapping natural language descriptions to computable formal graph representations.

## Citation
If you use this code or dataset in your research, please cite our paper:
*Citation details will be updated upon formal acceptance of the manuscript.*

## Note on Open Access
This repository currently includes the GNN training and inference code along with the full training dataset. The complete framework pipeline will be made publicly available upon acceptance of this paper.
