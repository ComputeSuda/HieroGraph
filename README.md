# HieroGraph
HieroGraph: The model, which consists of a knowledge-primed hierarchical heterogeneous graph and an iterative positive-unlabeled (PU) learning framework, comprehensively integrates pan-cancer multi-omics data across phosphosite, protein, and pathway levels to identify cancer genes and interpret their regulatory contexts.
![image](https://github.com/ComputeSuda/HieroGraph/blob/main/img/img.png)

# System requirement
HieroGraph is developed under Windows/Linux environment with:
* Python (3.9.25):
    - torch==2.8.0+cu126
    - torch-geometric==2.6.1
    - numpy==1.26.3
    - pandas==2.3.3
    - scipy==1.13.1
    - scikit-learn==1.6.1
    - captum==0.8.0
    - optuna==4.8.0
* You can install the core dependent packages by the following commands:
    - conda create -n hierograph python=3.9.25
    - conda activate hierograph
    - pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
    - pip install torch-geometric==2.6.1
    - pip install numpy==1.26.3 pandas==2.3.3 scipy==1.13.1 scikit-learn==1.6.1 captum==0.8.0 optuna==4.8.0

# Dataset
We provide a comprehensive multi-scale dataset curated from pan-cancer CPTAC cohorts spanning 10 cancer types, encompassing phosphoproteomics, proteomics, transcriptomics, copy number variation, and somatic mutation profiles.