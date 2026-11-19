# EGMP-PPIS
## 1 Description
    (a) EGMP-PPIS is a lightweight, single-stream equivariant geometry-aware message passing network for protein-protein interaction site (PPIS) prediction. It integrates evolutionary and structural residue features and formulates PPIS prediction as a residue-level binary node classification task on 3D spatial protein graphs.
    (b) The core component of EGMP-PPIS is the equivariant geometry-aware message-passing layer. Within a single parallelizable stream, it combines distance-aware message generation, semantic attention, E(n)-equivariant coordinate updates, and linear-complexity global context broadcasting, enabling the model to capture both local 3D geometric microenvironments and global protein topology simultaneously.
    (c) By introducing a prediction-stage residual fusion mechanism, the model concatenates the initial evolutionary embedding with the final graph representation prior to classification. This preserves complementary sequence information without extra graph propagation overhead and achieves strong performance on imbalanced bound and unbound PPIS benchmarks.

## 2 Installation
### 2.1 System Requirements
We recommend using a GPU for efficient training and prediction. To run EGMP-PPIS on GPU, please ensure the following requirement: CUDA >= 11.8.

### 2.2 Virtual Environment Dependencies
    (1) python 3.8
    (2) torch-2.1.2+cu118
    (3) torchaudio-2.1.2
    (4) torchvision-0.16.2
    (5) dgl_cu118.1.1.2
    (6) cudatoolkit-11.8
    (7) pandas
    (8) scikit-learn

## 3 Datasets
The folder `./Dataset/` contains datasets used in our experiments, including all PDB files, Test_315-28.pkl, Test_60.pkl, Train_335.pkl, UBtest_31-6.pkl, and bound_unbound_mapping31-6.txt. The remaining datasets are sourced from the original GraphPPIS. Large files are managed by Git LFS; please download via git clone and run git lfs install && git lfs pull, or [pdb and feature files](https://drive.google.com/drive/folders/1KMJjj7DPDJvVE2lF44gmEdGMpRj5sBQM?usp=drive_link) from Google Drive.

## 4 Features
    Extracted features are placed in the `./Feature/` directory. Detailed descriptions are as follows:
      (1) distance_map_SC: The residue side chain centroid is adopted as pseudo coordinates to calculate the distance matrix of protein chains.
      (2) dssp: DSSP matrix of the experimental protein chains.
      (3) hmm: HMM matrix of the experimental protein chains.
      (4) psepos: Pseudo coordinates of residues in the datasets. SC, CA and C denote side chain centroid, Cα atom and residue centroid, respectively.
      (5) pssm: PSSM matrix of the experimental protein chains.
      (6) resAF: Residue atom features of each protein.

## 5 The Trained Model
Pre-trained models are saved in the `./Model/` directory for quick evaluation.

## 6 Usage
The main architecture and ablation variants of the model are defined in EGMPPPIS_model.py.</br>
Run train.py to train the model from scratch.</br>
Run test.py to evaluate the trained model on benchmark test sets.
