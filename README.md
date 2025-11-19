# OBSERVE

**O**vert **B**ehavior **S**elf-supervised **E**stimation and **R**epresentation **V**ia **E**mbeddings

> A self-supervised learning framework for temporal representation learning from multimodal 
> facial behavioral features. This project explores SSL approaches for modeling overt attention 
> patterns without explicit frame-level labels.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

**OBSERVE** is a research framework investigating self-supervised representation learning 
for temporal behavioral sequences. Building upon the [Student Attention Analysis](https://github.com/[your-username]/student-attention-analysis) 
project, this work focuses on learning robust temporal representations from multimodal 
facial features through multi-task SSL objectives.

### Research Focus

This project demonstrates expertise in:
- **Self-Supervised Learning**: Multi-task SSL with contrastive, generative, and predictive objectives
- **Temporal Modeling**: Transformer-based architectures for behavioral sequence encoding
- **Representation Learning**: Learning attention-relevant embeddings without explicit labels
- **Multimodal Fusion**: Integrating gaze, expression, head pose, and affective signals

### Background

- **Institution:** Vellore Institute of Technology (VIT Chennai)
- **Duration:** June 2022 - February 2024
- **Collaboration:** Ministry of Electronics and IT, Government of India
- **Supervisors:** Dr. L. Jegannathan, Dr. Janaki Meena Murugan

---

## 🏗️ Framework Architecture

### Self-Supervised Learning Paradigm

OBSERVE employs a **Transformer-based encoder** trained with four complementary SSL objectives:

**1. Temporal Prediction**  
Learn future behavioral states from current context, capturing temporal dynamics of attention patterns.

**2. Behavioral Consistency**  
Enforce similar embeddings for similar attention trajectories through contrastive learning.

**3. Cross-Modal Alignment**  
Learn relationships between behavioral modalities (gaze ↔ expression ↔ engagement).

**4. Attention Flow Smoothness**  
Regularize temporal transitions to model natural behavioral dynamics.

### Model Specifications
```
Behavioral Transformer Encoder
├── Input: Temporal feature sequences (750 frames, 30 seconds)
├── Architecture: 8-layer Transformer
│   ├── Embedding Dimension: 768
│   ├── Attention Heads: 12
│   ├── Feedforward Dimension: 3072
│   └── Total Parameters: ~59M (~235 MB)
├── Training: Multi-task SSL objectives
└── Output: 768-dimensional temporal embeddings
```

**Training Hardware:** 2x NVIDIA Tesla T4 GPUs (16GB each)

---

## 🔬 Multimodal Feature Engineering

The framework processes three progressively refined feature sets, demonstrating systematic 
feature selection from comprehensive extraction to psychologically-grounded optimization.

### Feature Evolution

| Version | Features | Focus | Use Case |
|---------|----------|-------|----------|
| **v1** | 450 | Comprehensive extraction (all signals) | Exploratory analysis |
| **v2** | 176 | Correlation-based pruning (r > 0.95 removal) | Dimensionality reduction |
| **v3** | 79 | Psychological grounding (attention-relevant) | Final optimized |

### Feature Modalities (v3 - Final)

**Gaze & Eye Metrics** (18 features)
- Gaze direction (9-way classification)
- Eye aperture and openness
- Blink patterns (rate, duration, completeness)
- Fixation and saccade dynamics
- Pupil size statistics

**Head Pose & Movement** (15 features)
- 3D orientation (roll, pitch, yaw)
- Velocity and acceleration components
- Movement stability and jerk metrics
- Head-gaze coordination

**Facial Expression** (12 features)
- Action unit intensities
- Expression change rate
- Micro-expression frequency
- Facial asymmetry

**Affective State** (8 features)
- Valence and arousal estimates
- Emotion quadrant classification
- Affective stability measures

**Engagement Proxies** (26 features)
- Attention focus indicators
- Cognitive load indices
- Behavioral complexity
- Multimodal consistency scores

---

## 📦 Repository Contents

### Core Components
```
OBSERVE/
├── feature_extraction/
│   ├── raw_features.py              # Raw facial measurements & detections
│   ├── derived_features.py          # Temporal dynamics & contextual features
│   └── advanced_features.py         # Psychological state inference
│
├── notebooks/
│   ├── ssl-with-450-features.ipynb       # v1: Comprehensive features
│   ├── ssl-with-176-features.ipynb       # v2: Correlation-pruned
│   └── ssl-with-processed-features.ipynb # v3: Final optimized
│
├── models/
│   └── observe_best_79features.pth       # Pre-trained SSL encoder
│
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
└── README.md                              # This file
```

### Feature Extraction Pipeline

Hierarchical feature extraction from webcam-observable facial data:

**`raw_features.py` - Foundation Layer**
- Facial geometry (Eye Aspect Ratio, Mouth Aspect Ratio)
- Behavioral detections (blink, yawn, speech, drowsiness)
- Head pose & gaze direction (pitch, yaw, roll)
- Attention states (looking forward/away/down)
- CNN-based valence & arousal estimation (HydraNet)

**`derived_features.py` - Temporal Dynamics**
- Smile analysis (Duchenne score, authenticity)
- Facial asymmetry (bilateral AU comparison)
- Action Unit velocity & acceleration tracking
- Visual attention index & cognitive load
- Fatigue detection & micro-expression identification
- Granular emotion classification (joy, anger, sadness, etc.)

**`advanced_features.py` - Psychological Inference**
- Emotion suppression detection (macro vs. micro-expression mismatch)
- Temporal stability analysis (trends, peaks, valleys)
- Big Five personality traits assessment (OCEAN model)
- Engagement level decomposition (visual, emotional, cognitive)
- Stress & anxiety scoring
- Deception indicators (eye contact, micro-expressions, fidgeting)

*History window: 120 frames (~4 seconds at 30 FPS) for temporal feature computation*

### Jupyter Notebooks

Three notebooks document the iterative development process:

**1. `ssl-with-450-features.ipynb`**  
Initial exploration with comprehensive feature extraction. Demonstrates SSL training on 
high-dimensional temporal sequences using the full feature extraction pipeline.

**2. `ssl-with-176-features.ipynb`**  
Feature selection through correlation analysis. Shows dimensionality reduction while 
preserving information content by removing redundant features (r > 0.95).

**3. `ssl-with-processed-features.ipynb`**  
Final optimized version with psychologically-grounded feature selection (79 features). 
Includes complete SSL training pipeline with multi-task objectives and convergence analysis.

### Pre-trained Model

** `best_overall_model.pth`**  
- Pre-trained Behavioral Transformer encoder (v3 features)
- Trained on DAiSEE dataset (6,511 training videos)
- 768-dimensional embeddings per frame
- Multi-task SSL objectives (temporal, consistency, cross-modal, flow)
- Ready for downstream fine-tuning or embedding extraction

---

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
numpy>=1.24.0
opencv-python>=4.7.0
mediapipe>=0.9.0
scikit-learn>=1.2.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Load Pre-trained Model
```python
import torch
from models import BehavioralTransformer
from config import Config

# Load configuration
config = Config()

# Initialize model
model = BehavioralTransformer(config, actual_feature_dim=79)

# Load pre-trained weights
checkpoint = torch.load('models/observe_best_79features.pth', map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print("Pre-trained encoder loaded successfully")
```

### Extract Temporal Embeddings
```python
import pandas as pd

# Load processed features (750 frames x 79 features)
features_df = pd.read_csv('sample_features.csv')
features = torch.tensor(features_df.values).unsqueeze(0)  # Add batch dimension

# Extract embeddings
with torch.no_grad():
    embeddings = model(features, return_embeddings=True)
    # Shape: (1, 750, 768) - 768-d embedding per frame

print(f"Extracted embeddings: {embeddings.shape}")
print(f"Per-frame embedding dimension: {embeddings.shape[-1]}")
```

### Run Jupyter Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Open any of the three notebooks:
# - ssl-with-450-features.ipynb
# - ssl-with-176-features.ipynb  
# - ssl-with-processed-features.ipynb (recommended starting point)
```

---

## 🎓 Technical Deep Dive

### Self-Supervised Learning Objectives

The encoder is trained with a weighted combination of four SSL losses:
```
L_total = 2.0 × L_temporal + 0.5 × L_consistency + 
          0.4 × L_cross_modal + 0.3 × L_flow
```

**Temporal Prediction (Weight: 2.0)**
- Predict future behavioral states (50 frames ahead, ~2 seconds)
- Huber loss for robustness to outliers
- Primary learning signal

**Behavioral Consistency (Weight: 0.5)**
- Contrastive objective: similar attention patterns → similar embeddings
- MSE loss on embedding similarity vs. attention similarity matrices
- Encourages attention-relevant invariances

**Cross-Modal Alignment (Weight: 0.4)**
- Predict one modality from another (e.g., engagement from gaze)
- MSE loss on cross-modal predictions
- Captures multimodal relationships

**Attention Flow Smoothness (Weight: 0.3)**
- L1 penalty on temporal differences
- Regularizes abrupt transitions
- Encourages natural behavioral dynamics

### Training Configuration
```yaml
Model:
  - Architecture: Transformer Encoder (8 layers)
  - Embedding Dim: 768
  - Attention Heads: 12
  - Parameters: 58,719,055 (~59M)

Training:
  - Epochs: 250 (early stopping patience: 5)
  - Batch Size: 16 (8 per GPU)
  - Learning Rate: 2e-4 (OneCycleLR scheduler)
  - Optimizer: AdamW (weight_decay: 1e-4)
  - Mixed Precision: FP16 with gradient scaling
  - Gradient Clipping: max_norm = 1.0

Data:
  - Sequence Length: 750 frames (30 seconds at 25 FPS)
  - Train Videos: 6,511 (DAiSEE)
  - Test Videos: 1,720 (DAiSEE)
  - Augmentation: Temporal jitter, Gaussian noise, feature masking

Hardware:
  - GPUs: 2x NVIDIA Tesla T4 (16GB each)
  - Strategy: DataParallel
  - Training Duration: ~48 hours
```

---

## 📊 Dataset

This work uses the [DAiSEE (Dataset for Affective States in E-Environments)](https://people.iith.ac.in/vineethnb/resources/daisee/) 
dataset for training and evaluation.

**Dataset Characteristics:**
- 9,068 video clips from 112 participants
- Labels: Engagement, Boredom, Confusion, Frustration (0-3 scale)
- Context: Online learning sessions
- Split: 6,511 train / 1,720 test videos

**Citation:**
```bibtex
@inproceedings{gupta2016daisee,
  title={DAiSEE: Towards user engagement recognition in the wild},
  author={Gupta, Abhay and D'Cunha, Arjun and Awasthi, Kamal and Balasubramanian, Vineeth},
  booktitle={Proceedings of the 18th ACM International Conference on Multimodal Interaction},
  pages={121--128},
  year={2016}
}
```

**Access:** Dataset access requires registration at the DAiSEE website.

---

## 🎯 Applications & Use Cases

The learned temporal representations can be applied to various domains:

**Driver Safety Monitoring**
- Real-time distraction detection
- Drowsiness estimation from behavioral patterns
- Attention state assessment for ADAS

**Automated Proctoring**
- Engagement tracking during online examinations
- Behavioral anomaly detection
- Non-invasive monitoring for academic integrity

**Remote Learning Analytics**
- Student engagement measurement in online education
- Attention flow analysis across lecture segments
- Personalized learning interventions

**Human-Computer Interaction**
- Attention-aware interfaces
- Adaptive content delivery
- Cognitive load estimation for UX optimization

---

## 📚 Related Work & References

### Self-Supervised Learning

- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR), *ICML 2020*
- Grill et al., "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" (BYOL), *NeurIPS 2020*
- Yue et al., "TS2Vec: Towards Universal Representation of Time Series", *NeurIPS 2021*
- "Ti-MAE: Self-Supervised Masked Time Series Autoencoders", *arXiv 2023*

### Transformers for Sequences

- Vaswani et al., "Attention Is All You Need", *NeurIPS 2017*
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", *NAACL 2019*
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT), *ICLR 2021*

### Behavioral Analysis & Attention

- Ortubay et al., "Real-time estimation of overt attention from dynamic features of the face using deep-learning", 2023
- Baltrusaitis et al., "OpenFace 2.0: Facial Behavior Analysis Toolkit", *FG 2018*
- Mollahosseini et al., "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild", *ACII 2017*

### Dataset

- Gupta et al., "DAiSEE: Towards user engagement recognition in the wild", *ICMI 2016*

---

## 🙏 Acknowledgments

**Supervision:**
- **Dr. L. Jegannathan**,  Professor, VIT Chennai
- **Dr. Janaki Meena Murugan**, Professor, VIT Chennai

**Collaboration:**
- **Ministry of Electronics and Information Technology (MeitY)**, Government of India

**Institution:**
- **Vellore Institute of Technology (VIT)**, Chennai Campus

**Dataset:**
We thank the creators of the DAiSEE dataset for making their data available to the 
research community.

**Open Source Tools:**
This project builds upon excellent open-source frameworks:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) - Facial action unit detection
- [MediaPipe](https://google.github.io/mediapipe/) - Face mesh and iris tracking

---

## 📧 Contact

**Sarath Krishna Chingapurathu**  
Email: skchingapurathu@gmail.com  

### Open to Collaboration
I'm interested in research collaborations exploring self-supervised learning, temporal 
representation learning, and multimodal fusion. Feel free to reach out to discuss 
potential extensions of this work.

### Questions & Support
- **Implementation questions:** Open an issue on this repository
- **Research discussions:** Email me directly
- **Bug reports:** Use GitHub Issues

### Current Status
I'm currently applying to research Masters programs (MILA, Waterloo, UBC, Alberta, McGill) 
with interests in self-supervised learning and representation learning. Looking forward to 
continuing work in temporal modeling and multimodal learning with proper research mentorship 
and resources.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License
```
MIT License

Copyright (c) 2024 Sarath Krishna Chingapurathu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📖 Citation

If this work is useful for your research, please consider citing:
```bibtex
@misc{chingapurathu2024observe,
  author = {Chingapurathu, Sarath Krishna},
  title = {OBSERVE: Overt Behavior Self-supervised Estimation and Representation Via Embeddings},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/[your-username]/OBSERVE}},
  note = {Self-supervised learning framework for temporal behavioral representation learning. 
          Research conducted at VIT Chennai in collaboration with MeitY, Government of India.}
}
```

---

## 🔗 Related Projects

- [Student Attention Analysis](https://github.com/[your-username]/student-attention-analysis) - 
  Initial work on attention detection from facial features (CoCoNet'23)

---

**Built during undergraduate research at VIT Chennai (2022-2024)**

*This project represents exploration in self-supervised temporal representation learning, 
demonstrating systematic feature engineering, multi-task SSL training, and Transformer-based 
sequence modeling for behavioral analysis.*
