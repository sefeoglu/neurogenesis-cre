# Neurogenesis for Continual Relation Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-FFD21F.svg)](https://huggingface.co/docs/transformers/)
[![Status](https://img.shields.io/badge/Status-Research%20Project-orange.svg)](README.md)

This repository studies continual relation extraction with a BERT-based backbone, neurogenesis-inspired attention, and a mixture-of-tasks / mixture-of-experts classification layer. The goal is to support incremental learning across multiple relation-extraction tasks while preserving useful knowledge transfer and reducing destructive weight overwriting.

## Overview

The project combines:
- BERT-based relation classification
- neurogenesis-style random-feature attention adjustments
- dynamic task routing across continual tasks
- task-specific expert heads to encourage modular specialization
- parameter freezing for the shared encoder when training new tasks to reduce catastrophic forgetting

This is designed for continual or multi-task learning over TACRED-style relation extraction datasets.

## Repository structure

```bash
.
├── LICENSE
├── README.md
├── config.ini
├── data/
│   └── README.md
├── docs/
│   └── README.md
├── results/
│   └── README.md
├── src/
│   ├── data-preparation/
│   │   ├── datasplit_tacred.py
│   │   └── sampler.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── main.py
│   │   ├── model_transfer.py
│   │   ├── neuro_genesis.py
│   │   ├── re_dataset.py
│   │   ├── README.md
│   │   ├── test_bert.py
│   │   └── train_bert.py
│   ├── notebook/
│   │   ├── MixofTasks.ipynb
│   │   └── neurogenesis.ipynb
│   ├── significance_test/
│   │   └── T_Test_neurogenesis.ipynb
│   ├── text_entailment/
│   │   └── text_entailment.py
│   └── viz/
│       ├── README.md
│       └── plots.py
└──
```

## Main implementation

The core model logic lives in:
- [src/models/baseline.py](src/models/baseline.py)
- [src/models/train_bert.py](src/models/train_bert.py)
- [src/models/test_bert.py](src/models/test_bert.py)
- [src/models/main.py](src/models/main.py)

The exploratory notebook for the continual mixture-of-tasks design is in:
- [src/notebook/MixofTasks.ipynb](src/notebook/MixofTasks.ipynb)

## Requirements

Install the required Python dependencies before running the training pipeline:

```bash
python -m pip install -r requirements.txt
```

A minimal manual install is also possible:

```bash
pip install torch transformers scikit-learn tqdm numpy pandas scipy
```

For the notebook-based workflow, the environment can also require additional Hugging Face tooling and GPU support depending on the execution environment.

## Quick start

From the project root:

```bash
python src/models/main.py \
  --dataset-path /path/to/tacred/final \
  --output-dir ./results/neurogenesis \
  --phi performer \
  --epochs 5 \
  --batch-sizes 4
```

The script accepts:
- `--dataset-path`: TACRED-style dataset directory
- `--output-dir`: directory for saved checkpoints and metrics
- `--phi`: neurogenesis random-feature variant (`performer`, `cosine`, `linear`, etc.)
- `--epochs`: one or more epoch values
- `--batch-sizes`: one or more batch sizes

## Notes on the approach

- The shared BERT encoder is intentionally reused across tasks to support transfer.
- Task-specific heads and routing layers are kept relatively isolated to reduce uncontrolled overwriting of earlier task knowledge.
- The MoE-style output head is designed to selectively combine expert decisions rather than using a single flat classifier for all tasks.
- The notebook version contains the more experimental continual multi-task setup, while the model scripts provide the core training and evaluation flow.

## Citation / related work

The neurogenesis idea follows the work by Draelos et al.:

```bibtex
@inproceedings{draelos2017neurogenesis,
  author={Timothy J. Draelos and Nadine E. Miner and Christopher C. Lamb and Jonathan A. Cox and Craig M. Vineyard and Kristofor D. Carlson and William M. Severa and Conrad D. James and James B. Aimone},
  title={Neurogenesis Deep Learning: Extending Deep Networks to Accommodate New Classes},
  booktitle={2017 International Joint Conference on Neural Networks (IJCNN)},
  year={2017},
  pages={526--533},
  url={https://doi.org/10.1109/IJCNN.2017.7965898}
}
```

## License

This project is distributed under the license in [LICENSE](LICENSE).

> The code is intended for research and experimentation. Please cite the relevant work and acknowledge the repository appropriately when reusing or extending the implementation.
