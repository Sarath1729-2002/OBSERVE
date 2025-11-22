# TEMPO

**T**emporal **E**mbedding **M**odel for **P**atterns in **O**vert behavior

> A temporal representation learning framework for modeling behavioral sequences from multimodal 
> facial features. This project explores Transformer-based approaches for learning attention-relevant 
> temporal patterns from webcam-observable behavioral data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

**TEMPO** is a research framework for learning temporal representations from behavioral sequences. 
Building upon initial work on attention detection ([CoCoNet'22](link-to-paper)), this project 
focuses on learning robust temporal embeddings from multimodal facial features through predictive 
modeling of behavioral dynamics.

### Research Focus

This project demonstrates:
- **Temporal Representation Learning**: Predicting future behavioral states to capture temporal dynamics
- **Transformer-Based Sequence Modeling**: 8-layer encoder for behavioral sequences (30-second windows)
- **Systematic Feature Engineering**: 450 → 176 → 79 feature reduction with psychological grounding
- **Multi-Task Learning**: Temporal prediction with auxiliary objectives for behavioral consistency

### Context

- **Institution:** Vellore Institute of Technology (VIT Chennai)
- **Duration:** June 2022 - February 2024
- **Collaboration:** Ministry of Electronics and IT, Government of India (Sanction: L-14011/4/2022-HRD)
- **Supervisors:** Prof. Dr. L. Jegannathan, Prof. Dr. Janaki Meena Murugan
- **Initial Publication:** CoCoNet 2022 Conference (attention monitoring)

---

## 🏗️ Framework Architecture

### Temporal Learning Paradigm

TEMPO employs a **Transformer-based encoder** trained with multiple temporal objectives:

**1. Future State Prediction (Primary Objective)**  
Learn to predict behavioral states 2 seconds into the future from current context. This temporal 
prediction task serves as the main learning signal, forcing the model to capture attention-relevant 
dynamics without explicit frame-level labels.

**2. Behavioral Consistency Regularization**  
Encourage stable embeddings for similar attention patterns through consistency constraints on the 
learned representations.

**3. Cross-Modal Feature Alignment**  
Learn relationships between behavioral modalities (gaze ↔ expression ↔ engagement) by predicting 
one feature subset from another.

**4. Temporal Smoothness Regularization**  
Penalize abrupt transitions in the learned attention flow to encourage natural behavioral dynamics.

### Model Specifications
```
Behavioral Transformer Encoder
├── Input: Temporal feature sequences (750 frames, 30 seconds)
├── Architecture: 8-layer Transformer
│   ├── Embedding Dimension: 768
│   ├── Attention Heads: 12
│   ├── Feedforward Dimension: 3072
│   └── Total Parameters: ~59M (~235 MB)
├── Training: Multi-task temporal objectives
└── Output: 768-dimensional temporal embeddings
```

**Training Hardware:** 2x NVIDIA Tesla T4 GPUs (16GB each)  
**Design Rationale:** Architecture sized for deployment on consumer hardware while maintaining 
representational capacity for complex behavioral patterns.

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

**Key Insight:** Systematic ablation studies (documented in notebooks) show that psychologically-
grounded feature selection (v3) maintains representational quality while enabling efficient 
training and inference on resource-constrained hardware.

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
TEMPO/
├── feature_extraction/
│   ├── raw_features.py              # Raw facial measurements & detections
│   ├── derived_features.py          # Temporal dynamics & contextual features
│   └── advanced_features.py         # Psychological state inference
│
├── notebooks/
│   ├── temporal_learning_v1_450_features.ipynb  # v1: Comprehensive features
│   ├── temporal_learning_v2_176_features.ipynb  # v2: Correlation-pruned
│   └── temporal_learning_v3_79_features.ipynb   # v3: Final optimized
│
├── models/
│   └── best_overall_model.pth          # trained weights
│
├── requirements.txt                              # Python dependencies
├── LICENSE                                       # MIT License
└── README.md                                     # This file
```

### Feature Extraction Pipeline

Hierarchical feature extraction from webcam-observable facial data:

**`raw_features.py` - Foundation Layer**
- Facial geometry (Eye Aspect Ratio, Mouth Aspect Ratio)
- Behavioral detections (blink, yawn, speech, drowsiness)
- Head pose & gaze direction (pitch, yaw, roll)
- Attention states (looking forward/away/down)
- CNN-based valence & arousal estimation (ResNet-18 HydraNet)

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

**1. `temporal_learning_v1_450_features.ipynb`**  
Initial exploration with comprehensive feature extraction (450 features). Demonstrates baseline 
temporal modeling on high-dimensional sequences using the full feature extraction pipeline.

**2. `temporal_learning_v2_176_features.ipynb`**  
Feature selection through correlation analysis (r > 0.95 threshold). Shows dimensionality 
reduction while preserving information content by removing redundant features.

**3. `temporal_learning_v3_79_features.ipynb`** ⭐  
Final optimized version with psychologically-grounded feature selection (79 features). 
Includes complete training pipeline with multi-task temporal objectives, convergence analysis, 
and downstream evaluation on attention prediction task.

### Pre-trained Model

**`temporal_encoder_79features.pth`**  
- Pre-trained Behavioral Transformer encoder (v3 features)
- Trained on DAiSEE dataset (6,511 training videos)
- 768-dimensional embeddings per frame
- Multi-task temporal objectives (prediction, consistency, alignment, smoothness)
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
checkpoint = torch.load('models/temporal_encoder_79features.pth', map_location='cuda')
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
# - temporal_learning_v1_450_features.ipynb
# - temporal_learning_v2_176_features.ipynb  
# - temporal_learning_v3_79_features.ipynb (recommended starting point)
```

---

## 🎓 Technical Deep Dive

### Training Objectives

The encoder is trained with a weighted combination of temporal learning objectives:
```
L_total = 2.0 × L_temporal + 0.5 × L_consistency + 
          0.4 × L_cross_modal + 0.3 × L_smoothness
```

**Temporal Prediction (Weight: 2.0)**
- Predict future behavioral states (50 frames ahead, ~2 seconds)
- Huber loss for robustness to outliers
- Primary learning signal - forces model to capture attention dynamics

**Behavioral Consistency (Weight: 0.5)**
- Regularization objective: similar attention patterns → similar embeddings
- MSE loss on embedding similarity vs. attention similarity matrices
- Encourages attention-relevant invariances in learned representations

**Cross-Modal Alignment (Weight: 0.4)**
- Predict engagement features from gaze/expression features
- MSE loss on cross-modal predictions
- Captures multimodal relationships without explicit fusion

**Temporal Smoothness (Weight: 0.3)**
- L1 penalty on temporal differences in learned attention flow
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

## 📊 Results & Evaluation

### Downstream Task: Attention Prediction

After pre-training with temporal objectives, the encoder was fine-tuned for attention prediction 
using video-level proxy scores derived via Analytic Hierarchy Process (AHP) from DAiSEE labels 
(Engagement, Boredom, Confusion, Frustration).

**Attention Proxy Score Formula:**
```
Attention = (E - 0.29×B - 0.17×C - 0.54×F) / 3
```
Weights derived from AHP prioritization matrices based on psychological attention literature.

**Performance Metrics (Test Set - 1,577 videos):**
```
Mean Squared Error (MSE):     0.0096
Root Mean Squared Error:      0.0981
Mean Absolute Error (MAE):    0.0791
R² Score:                     0.0294
Pearson Correlation:          0.2154 (p < 0.001)
```

### Interpretation

The modest downstream performance (R² = 0.03) reveals important insights about attention modeling 
from facial features:

1. **Label Noise**: DAiSEE annotations have known inter-annotator variability (~0.6 agreement), 
   limiting the upper bound on correlation with proxy scores.

2. **Temporal Granularity**: 30-second windows may be too coarse for capturing local attention 
   shifts, which often occur at sub-5-second scales.

3. **Modality Limitations**: Webcam-observable facial features provide weaker signal than 
   eye-tracking or physiological measures. The correlation of 0.22 suggests facial cues capture 
   only partial aspects of attention.

4. **Representation Quality**: The learned embeddings capture temporal patterns (as evidenced 
   by convergent temporal prediction loss during pre-training), but these patterns may reflect 
   general behavioral dynamics rather than attention-specific features.

### Future Directions

Based on these results, promising directions include:
- **Multi-modal fusion**: Incorporating audio (speech patterns, silence) and screen interaction data
- **Hierarchical temporal modeling**: Learning at multiple timescales (1s, 5s, 30s windows)
- **Eye-tracking integration**: Fine-grained gaze data for stronger supervision signal
- **Contrastive pre-training**: Proper SimCLR/MoCo implementation with data augmentation strategies
- **Larger-scale pre-training**: More diverse behavioral data beyond online learning contexts

---

## 📊 Dataset

This work uses the [DAiSEE (Dataset for Affective States in E-Environments)](https://people.iith.ac.in/vineethnb/resources/daisee/) 
dataset for training and evaluation.

**Dataset Characteristics:**
- 9,068 video clips from 112 participants
- Labels: Engagement, Boredom, Confusion, Frustration (0-3 scale)
- Context: Online learning sessions (MOOC lectures)
- Split: 6,511 train / 1,720 test videos
- Duration: 10 seconds per clip (variable FPS)

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

**Remote Learning Analytics**
- Student engagement measurement in online education
- Attention flow analysis across lecture segments
- Personalized learning interventions based on behavioral patterns

**Human-Computer Interaction**
- Attention-aware interfaces that adapt to user cognitive load
- Cognitive load estimation for UX optimization
- Adaptive content delivery based on engagement signals

**Driver Safety Monitoring**
- Real-time distraction detection from facial cues
- Drowsiness estimation from behavioral patterns
- Attention state assessment for ADAS systems

**Automated Proctoring**
- Engagement tracking during online examinations
- Behavioral anomaly detection
- Non-invasive monitoring for academic integrity

---

## 📚 Related Work & References

### Temporal Representation Learning

- Yue et al., "TS2Vec: Towards Universal Representation of Time Series", *NeurIPS 2021*
- Eldele et al., "Time-Series Representation Learning via Temporal and Contextual Contrasting", *IJCAI 2021*
- Franceschi et al., "Unsupervised Scalable Representation Learning for Multivariate Time Series", *NeurIPS 2019*

### Self-Supervised Learning (Context)

- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR), *ICML 2020*
- Grill et al., "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" (BYOL), *NeurIPS 2020*
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE), *CVPR 2022*

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
- **Prof. Dr. L. Jegannathan**, Professor, VIT Chennai
- **Prof. Dr. Janaki Meena Murugan**, Professor, VIT Chennai

**Funding:**
- **Ministry of Electronics and Information Technology (MeitY)**, Government of India  
  Sanction No: L-14011/4/2022-HRD

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
I'm interested in research collaborations exploring temporal representation learning, 
sequence modeling, and multimodal behavioral analysis. Feel free to reach out to discuss 
potential extensions of this work.

### Questions & Support
- **Implementation questions:** Open an issue on this repository
- **Research discussions:** Email me directly
- **Bug reports:** Use GitHub Issues

### Current Status
I'm currently applying to research Masters programs (MILA, Waterloo, UBC, Alberta, McGill) 
with interests in temporal modeling, representation learning, and AI reasoning. Looking forward 
to continuing work in sequence understanding and behavioral modeling with proper research 
mentorship and resources.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If this work is useful for your research, please consider citing:
```bibtex
@misc{chingapurathu2024tempo,
  author = {Chingapurathu, Sarath Krishna},
  title = {TEMPO: Temporal Embedding Model for Patterns in Overt Behavior},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/[your-username]/TEMPO}},
  note = {Temporal representation learning framework for behavioral sequence modeling. 
          Research conducted at VIT Chennai in collaboration with MeitY, Government of India.}
}
```

---

## 🔗 Related Projects

- [Student Attention Analysis](https://github.com/[your-username]/student-attention-analysis) - 
  Initial work on attention detection from facial features (CoCoNet'22)

---

**Built during undergraduate research at VIT Chennai (2022-2024)**

*This project explores temporal representation learning for behavioral analysis, demonstrating 
systematic feature engineering, multi-task training, and Transformer-based sequence modeling 
under resource constraints. Results highlight both the potential and limitations of learning 
attention patterns from facial features alone.*
