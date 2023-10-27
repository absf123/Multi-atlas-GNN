# Spectral Graph Neural Network-based Multi-atlas Brain Network Fusion for Major Depressive Disorder Diagnosis



## Overview

We propose a novel multi-atlas fusion method that incorporates early and late fusion in a unified framework. Our method introduces the concept of the holistic Functional Connectivity Network (FCN), which captures both intra-atlas relationships within individual atlases and inter-regional relationships between atlases with different brain parcellation scales.

The holistic Functional Connectivity Network (FCN) captures both intra-atlas relationships within individual atlases and inter-regional relationships between atlases with different brain parcellation scales. This comprehensive representation enables the identification of potential disease-related patterns associated with MDD in the early stage of our framework.

## Code list

- Config.py
- data.py : load data
- model.py : gnn based model architecture
- main.py : single atlas & Holistic altas (Early fusion)
- main_Late_fusion.py : multiple atlases Late fusion
- main_Ours.py : multiple atlases Ours
- train.py : 'main' training code
- train_Late_fusion.py : 'main_Late_fusion' training code
- train_Ours.py : 'main_Ours' training code
- graph_utils.py : graph model function
- utils.py : acc,sen,spec function
- Make_FC_data.py : make functional connectivity data including Holistic FCN
- ttest.py : group-level ttest
- etc


## Requirements

To run this project, you will need:
- Python 3.9.7 or higher
- torch 1.11.0+cu113
- numpy 1.22.4
- scikit-learn 1.0.2
- etc.
  

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2019-0-00079, Artificial Intelligence Graduate School Program(Korea University), No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration), and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS202300212498). (Corresponding author: T.-E. Kam.)


