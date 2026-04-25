# DeepChem AlphaFold Integration

Prototype workflows for integrating AlphaFold protein structure data into DeepChem machine learning pipelines.

## Overview

This repository explores how predicted protein structures from AlphaFold can be converted into usable features for DeepChem models.

The project was inspired by the DeepChem feature request:

**Add support for AlphaFold model integration (#4965)**

## Goals

* Retrieve AlphaFold structures using UniProt IDs
* Parse PDB files and extract structural signals
* Use pLDDT confidence scores as learning features
* Build baseline protein classification pipelines in DeepChem
* Benchmark broad vs family-specific protein tasks
* Explore future residue-level and graph-based models

## Implemented Features

### AlphaFold Retrieval

* Automatic download of AlphaFold structures from AlphaFold DB
* UniProt ID based workflows

### Protein Featurization

Current prototype features include:

* sequence length
* amino acid composition
* cysteine / histidine fractions
* motif counts
* mean pLDDT confidence
* low-confidence residue ratio
* coarse structural descriptors

### DeepChem Modeling

Baseline models built with DeepChem:

* MultitaskClassifier
* rapid prototype training pipelines
* small benchmark datasets

## Benchmarks

### 1. Broad DNA-Binding Classification

General DNA-binding proteins vs non-DNA proteins.

**Observation:** coarse global features were insufficient for strong discrimination.

### 2. Zinc Finger Family Classification

Zinc finger proteins vs control proteins.

**Observation:** family-specific tasks showed stronger signal than broad functional grouping.

## Preliminary Results

| Task                    | Accuracy | F1 Score |
| ----------------------- | -------- | -------- |
| Broad DNA-binding       | 0.50     | 0.40     |
| DNA (improved features) | 0.50     | 0.67     |
| Zinc Finger Family      | 0.50     | 0.57     |

These results suggest that richer local residue features and graph representations are likely needed.

## Repository Structure

```text
.
├── src/           # Python scripts
├── data/          # CSV benchmark datasets
├── structures/    # Example AlphaFold PDB files
├── notebooks/     # Colab experiments
├── results/       # Logs / metrics
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Example Usage

### Download AlphaFold Structure

```bash
python src/download_alphafold_api.py P69905
```

### Run Zinc Finger Benchmark

```bash
python src/zinc_benchmark.py
```

## Future Work

* residue-level featurizers
* contact-map generation
* protein graph neural networks
* ESM / ProtBERT embeddings
* larger curated benchmark datasets
* native DeepChem featurizer integration

## Why This Matters

DeepChem has strong chemistry tooling. Protein structure support through AlphaFold-style workflows can expand applications in:

* target biology
* protein property prediction
* structure-aware screening
* multimodal drug discovery

## Status

Exploratory research prototype with working baselines and active development.

## Author

Prakhya Khandelwal

