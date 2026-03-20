# SeD-UD

**An Influence-Driven and Hierarchically-Decoupled Information Bottleneck for Multimodal Intent Recognition**

---

## Introduction

Multimodal intent recognition (MIR) is hindered by substantial redundancy and noise originating from text, speech, and visual inputs, which weakens feature distinctiveness and ultimately harms recognition performance. Although recent approaches based on the information bottleneck (IB) principle mitigate this issue via feature compression and reconstruction to obtain compact and noise-reduced representations, they still encounter two major drawbacks. First, conventional IB employs a fixed bottleneck dimension, making it unable to accommodate sample-dependent variations in redundancy and noise. Second, simultaneously handling redundancy and noise within a single compression process leads to incomplete feature purification. In this paper, we propose a novel framework named SeD-UD, which incorporates influence-driven input-adaptive bottleneck (IDAB) modules following a hierarchically-decoupled IB structure. Given a redundancy/noise influence factor, IDAB dynamically adjusts dimensions and selects the optimal parameters for compression and reconstruction, thereby achieving the best trade-off between information preservation and interference suppression. The IB structure performs hierarchically-decoupled processing of redundancy and noise via separated de-redundancy and unified denoising based on IDAB modules. Extensive experiments on benchmark datasets show SeD-UD outperforms current state-of-the-art models.

Multimodal intent recognition (MIR) is hindered by substantial redundancy and noise originating from text, speech, and visual inputs, which weakens feature distinctiveness and ultimately harms recognition performance. SeD-UD is a novel framework that incorporates **Influence-Driven Input-Adaptive Bottleneck (IDAB)** modules following a **hierarchically-decoupled information bottleneck strategy**.

Given a redundancy/noise influence factor, IDAB dynamically adjusts dimensions and selects the optimal parameters for compression and reconstruction, thereby achieving the best trade-off between information preservation and interference suppression. The framework performs hierarchically-decoupled processing of redundancy and noise via **separated de-redundancy** and **unified denoising** based on IDAB modules.

---

## Project Structure

```
SeD-UD/
├── main.py                 # Main training and evaluation script
├── configs-MIntRec.py      # Configuration for MIntRec dataset
├── configs-MELD-DA.py      # Configuration for MELD-DA dataset
├── requirements.txt        # Dependencies
├── data/
│   ├── __init__.py         # Dataset configurations
│   ├── base.py             # Data manager
│   ├── BERTencoder.py      # BERT text encoder
│   ├── mm_pre.py           # Multimodal preprocessing
│   ├── text_pre.py         # Text preprocessing
│   └── utils.py            # Data utilities
├── losses/
│   ├── __init__.py
│   └── total_loss.py       # Total loss computation (classification + saliency)
└── utils/
    ├── __init__.py
    ├── alignment.py        # Feature alignment
    └── Function.py         # Utility functions
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (recommended for GPU acceleration)

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
transformers>=4.35.0
matplotlib>=3.7.0
tensorboard>=2.14.0
pillow>=10.0.0
tqdm>=4.66.0
```

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/9meiye/SeD-UD.git
cd SeD-UD
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download pre-trained BERT model

Download `bert-base-uncased` from [Hugging Face](https://huggingface.co/bert-base-uncased) and update the path in the corresponding config file (e.g., `configs-MIntRec.py`).

---

## Dataset Configuration

### Supported Datasets

| Dataset | Samples | Labels | Description |
|---------|---------|--------|-------------|
| MIntRec | 2,224 | 20 | Multimodal Intent Recognition with fine-grained categories (11 for attitudes, 9 for goals) |
| MELD-DA | 9,989 | 12 | Emotion-related intents in multi-person conversation scenarios |

### Configuration

The project uses dataset-specific configuration files. Update parameters in the corresponding config file:

- `configs-MIntRec.py` - Configuration for MIntRec dataset
- `configs-MELD-DA.py` - Configuration for MELD-DA dataset

---

## Usage

### Training and Evaluation

```bash
python main.py
```

---

## Results

The model outputs comprehensive evaluation metrics:

- Accuracy (ACC)
- Weighted Precision
- Weighted Recall
- Weighted F1 Score

---

## License

This project is licensed under the MIT License.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{SeD-UD,
  title = {SeD-UD: An Influence-Driven and Hierarchically-Decoupled Information Bottleneck for Multimodal Intent Recognition},
  author = {Anonymous},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2026}
}
```

---

## Contact

For questions and issues, please open an issue on GitHub.
