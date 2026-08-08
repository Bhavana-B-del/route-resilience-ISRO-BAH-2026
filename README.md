# Route Resilience: PathMamba for Remote Sensing Road Extraction under Canopy and Occlusion

Official repository for **Team Madras Intelligence**'s submission to the NRSC-ISRO Grand Finale Hackathon.

---

## Performance and Hackathon Achievements

* **Selection Ratio:** Shortlisted as one of 34 finalist teams selected from a nationwide pool of over 15,000+ competing teams.
* **Final Rank:** Achieved a **Top-9 finish** in the Grand Finale following technical presentations.
* **Executive Presentation:** Presented model architecture, occlusion-handling methodology, and inference performance directly before the Director of NRSC-ISRO and senior leadership from IIT Hyderabad.

---

## Project Overview

Route Resilience addresses the critical challenge of extracting road networks from high-resolution satellite imagery under severe environmental occlusions such as dense tree canopies, shadows, and cloud cover. Standard convolutional segmentation models frequently disconnect road topologies when spectral signals are blocked. 

Our solution, **PathMamba**, integrates multi-angle Gray-Level Co-occurrence Matrix (GLCM) texture variance with dual State Space Model (Mamba) blocks to perform long-range spatial reasoning, bridging visual gaps in occluded infrastructure.

---

## Model Architecture

PathMamba utilizes a four-channel hybrid architecture designed for joint feature extraction and topological continuity:

1. **Input Representation (4 Channels):**
   * **RGB Channels (3):** Standard optical imagery normalized per-channel.
   * **GLCM Occlusion Channel (1):** Real-time, vectorized multi-angle texture computation evaluating local contrast and homogeneity to explicitly highlight regions with high occlusion potential.

2. **Feature Encoder:**
   * **Backbone:** `tf_efficientnet_b6.ns_jft_in1k` initialized with JFT-300M pre-training weights.
   * **Adapted Input Conv:** Modified stem convolution accepting 4 input channels with normalized weight distribution.

3. **Bottleneck & Sequence Modeling:**
   * **Stacked Mamba Blocks:** Dual sequential State Space Model (SSM) layers (`d_model` derived from feature channels, `d_state=16`, `d_conv=4`) utilizing pre-norm residual connections to sequence long-range spatial context.
   * **Cross-Attention:** Multi-head attention layer (8 heads) refining feature correlations across the flattened spatial sequence.

4. **Decoder & Output Heads:**
   * **UNet-Style Decoder:** Transpose-convolutional feature reconstruction with skip connections from encoder stages.
   * **Road Mask Head:** Primary 1-channel output predicting binary road segmentation.
   * **Confidence Head:** Secondary 1-channel output estimating model confidence specifically across occluded regions.

---

## Loss Function Design

The framework utilizes a multi-term combined loss function optimized for precision, topological accuracy, and edge definition:

* **Tversky Loss:** Weighted towards precision ($\alpha=0.7, \beta=0.3$) to eliminate boundary over-expansion.
* **Boundary Loss:** Distance-transform weighted loss targeting sharp road edge definition.
* **clDice Loss:** Soft-skeletonization loss preserving network connectivity and centerline continuity.
* **Buffered IoU Loss:** Max-pooling dilated IoU loss accommodating 3 to 5 pixel positioning tolerances.
* **Occlusion-Weighted Confidence Loss:** Direct supervision of the confidence head using ground truth masks weighted by GLCM texture response.

---

## Datasets and Training Pipeline

### Pre-Training Phase (UrbanMix)
The foundational feature representations were trained across a diverse combined dataset comprising multiple spatial resolutions and geographical terrain types:
* **Massachusetts Roads Dataset**
* **DeepGlobe Road Extraction Dataset**
* **SpaceNet-3 & SpaceNet-5 Datasets**
* **UrbanMix Binary Dataset**

### Fine-Tuning Phase (Cartosat-3)
* **Cartosat-3:** Domain-specific fine-tuning using high-resolution Cartosat-3 satellite imagery provided by ISRO.

### Data Augmentations
To enforce occlusion invariance, the pipeline incorporates specialized augmentations:
* **Canopy Patch Pasting:** Dynamic insertion of high-texture vegetation patches over valid road regions.
* **Thin-Road Snippet Injection:** Synthetic placement of narrow road segments derived from distance transform metrics.
* **Cloud & Shadow Simulation:** Procedural generation of soft, color-shifted atmospheric occlusions and directional shadows.
* **Cutout & Contrast Jittering:** Random background masking and per-image illumination variance.

---

## Model Weights

Model checkpoint weights are available upon request for research and validation purposes. To obtain access to trained weights (including pre-trained Stage A and Cartosat-3 fine-tuned checkpoints), please contact the team at:

**Email:** `bhavana4384@gmail.com`

## Team & Mentorship
Team Madras Intelligence :
* Abhinav
* Vishal
* Mahir Ali
* Bhavana

### Mentorship & Special Thanks
We express our gratitude to our mentors from the Indian Space Research Organisation (ISRO) for their guidance throughout the competition:

* Uday Kumar (ISRO Scientist)
* Pruthvi Raj (ISRO Scientist)
* Mayukh Mukherjee (ISRO Scientist)
---
