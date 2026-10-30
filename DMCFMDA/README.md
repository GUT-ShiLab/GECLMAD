# DMCFMDA
DMCFMDA：A Dual-Channel Multi-Source Cross-Modal Fusion Network with Contrastive Learning for Microbe-Disease Prediction

## Requirements
  * dgl-cu102==0.6.1
  * networkx==2.5
  * numpy==1.21.5
  * scikit-learn==1.0.2
  * torch==1.12.0+cu102
  * tqdm==4.66.4

## File
### Data
This section provides details about the datasets used in the project, including HMDAD, Disbiome, and FGMDI. Each dataset contains the following folders and files:
1. microbe_features.txt
This folder contains pre-aggregated similarity features of microbes. These features represent the relationships and similarities between different microbes, which have been computed using established methods.
2. disease_features.txt
This folder includes pre-aggregated similarity features of diseases. Similar to microbe_features, these features represent the relationships and similarities between different diseases.
3. adj.txt
This file represents the known associations between microbes and diseases in the dataset. It is a binary adjacency matrix, where:
Rows correspond to microbes.
Columns correspond to diseases.
Values are 1 for known associations and 0 otherwise.

4. interaction
This file represents the interaction matrix for microbe-disease associations. It captures the numerical values or scores related to the associations, which may be used as input to models or as ground truth for evaluation.


### code
  * eval.py: The startup code of the program
  * train.py: Train the model
  * DMCFMDA.py: Structure of the model
  * utils.py: Methods of data processing
 
## Usage
  * download code and data
  * execute ```python eval.py```

