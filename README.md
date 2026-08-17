# Disease GNN Relation Prediction

> Graph Machine Learning for Chemical-Disease Relation (CDR) prediction using GraphSAGE and PyTorch Geometric on biomedical literature graphs.

[Repository](https://github.com/SanyogSingh07/Disease-GNN-Relation-Prediction)

---

## Overview

**Disease GNN** is a Graph Machine Learning project that models complex biological relationships between chemical compounds and diseases. By constructing a heterogeneous biomedical entity graph from PubMed text corpora (BC5CDR dataset), the system uses an inductive Graph Neural Network (GraphSAGE) to predict novel Chemical-Disease Relations (CDR).

---

## Problem

Understanding how chemical compounds interact with diseases is fundamental to drug discovery and repurposing. Extracting these relationships from unstructured PubMed abstracts faces key challenges:
- **Relational Complexity**: Traditional NLP models evaluate entity pairs in isolation, ignoring broader network topology across literature.
- **Inductive Generalization**: Biomedical graphs continuously expand as new research papers and entities are published.

---

## Graph Construction & Methodology

```
[ PubMed BC5CDR Abstracts ] ──► [ PubTator NER Parsing ]
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │           GRAPH CONSTRUCTION          │
                     │  Nodes: Chemicals & Diseases          │
                     │  Edges: Co-occurrence & Relationships │
                     │  Features: TF-IDF & BioEmbeddings    │
                     └───────────────────────────────────────┘
                                         │
                                         ▼
                             [ GraphSAGE Conv Layers ]
                                         │
                                         ▼
                            [ Edge Link Prediction ]
```

### 1. Node & Edge Construction
- **Nodes ($V$)**: Represent distinct chemical compounds (e.g., *Acetaminophen*) and disease entities (e.g., *Hepatic Necrosis*).
- **Edges ($E$)**: Represent co-occurrence and annotated interactions within literature abstracts.
- **Node Features ($X$)**: Formulated via TF-IDF vectorization and semantic domain embeddings.

### 2. GNN Architecture (GraphSAGE)
- **Aggregator**: Mean aggregation over local 2-hop graph neighborhoods.
- **Inductive Representation**: Generates embeddings for previously unseen chemical/disease nodes without retraining the full graph topology:
  $$h_v^{(k)} = \sigma \left( W \cdot \text{CONCAT} \left( h_v^{(k-1)}, \text{AGGREGATE}_k \left( \{ h_u^{(k-1)}, \forall u \in \mathcal{N}(v) \} \right) \right) \right)$$

---

## Evaluation & Results

> [!NOTE]
> Link prediction evaluation metrics are computed on the BC5CDR test split.

| Metric | Target / Score |
|:---|:---|
| **ROC-AUC (Link Prediction)** | Evaluated on BC5CDR test set |
| **Average Precision (AP)** | Evaluated on BC5CDR test set |
| **Graph Scaling** | Tested up to 10k+ nodes & 50k+ edges |

---

## Project Structure

```
Disease-GNN-Relation-Prediction/
├── README.md
├── requirements.txt
├── main.py                # Pipeline Execution Script
├── src/
│   ├── graph_builder.py   # PubTator Parser & Graph Construction
│   ├── gnn_model.py       # PyTorch Geometric GraphSAGE Architecture
│   └── train_eval.py      # Link Prediction Training Loop
└── data/                  # BC5CDR Dataset Files
```

---

## Installation & Usage

```bash
git clone https://github.com/SanyogSingh07/Disease-GNN-Relation-Prediction.git
cd Disease-GNN-Relation-Prediction
pip install -r requirements.txt

# Run Graph Construction and GNN Training
python main.py
```

---

## Limitations & Educational Disclaimer

- **Limitations**: Current node features rely on textual TF-IDF co-occurrence; integrating molecular SMILES structures and 3D protein targets is planned.
- **Disclaimer**: This is an academic research demonstration designed to explore graph representation learning and is not intended for clinical pharmacology.
