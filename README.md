# RoadDamage-XAI

### Explainable Road Damage Classification Using Transfer Learning and Grad-CAM

RoadDamage-XAI is a computer vision project that classifies road damage into **Pothole, Crack, and Manhole** categories using **MobileNetV2 transfer learning** and explains the model's predictions using **Grad-CAM**.

The project starts with a road-damage dataset containing YOLO polygon annotations. Instead of directly training an object detector, the annotated regions are converted into individual image crops and used to build a three-class image classification system.

The main goal is not only to answer:

> **What type of road damage is present?**

but also:

> **Which part of the image influenced the model's decision?**

---

## Problem Statement

Road-condition monitoring is an important computer vision problem because manually inspecting large numbers of road images can be time-consuming and difficult to scale.

A classification model can help identify different types of road damage automatically. However, a prediction alone does not explain why the model made that decision. This can make computer vision systems harder to analyse and trust, particularly when the model makes an incorrect prediction.

This project addresses the problem by developing a lightweight image-classification pipeline that:

1. Extracts annotated road-damage regions from YOLO polygon labels.
2. Classifies each region as **Pothole, Crack, or Manhole**.
3. Provides a confidence score for the prediction.
4. Uses **Grad-CAM** to visualise the image regions that contributed to the prediction.

The project therefore combines **computer vision, transfer learning, model evaluation, and explainable AI** in a single workflow.

---

## Abstract

This project presents an explainable deep-learning approach for classifying road-damage images into three categories: **Pothole, Crack, and Manhole**.

The original dataset contains 2,009 road images with 4,737 annotated damage instances represented using YOLO polygon annotations. These annotations were parsed and converted into individual damage crops using padded bounding boxes. To reduce the risk of data leakage, the dataset was split at the **original-image level**, ensuring that crops originating from the same source image remained within the same data split.

A **MobileNetV2** model pretrained on ImageNet was used as the feature extractor. A classification head consisting of global average pooling, dropout, and a softmax output layer was first trained with the pretrained backbone frozen. The upper layers of MobileNetV2 were then fine-tuned using a smaller learning rate.

On the held-out test set, the final model achieved **69.51% accuracy** and a **63.33% macro F1-score**. Crack classification performed substantially better than Pothole and Manhole classification.

To improve interpretability, **Grad-CAM** was applied to the final convolutional layer of MobileNetV2. This produces heatmaps showing the regions that contributed most strongly to the predicted class.

The project demonstrates a practical pipeline for combining **transfer learning with explainability** for road-condition image analysis.

---

## Dataset

The project uses the **Road Damage Dataset: Potholes, Cracks and Manholes**.

### Dataset summary

| Property                 |     Value |
| ------------------------ | --------: |
| Original images          |     2,009 |
| Annotated objects        |     4,737 |
| Extracted crops          |     4,736 |
| Number of classes        |         3 |
| Input size               | 224 × 224 |
| Training source images   |     1,406 |
| Validation source images |       301 |
| Test source images       |       302 |

### Classes

| Class   | Annotated instances |
| ------- | ------------------: |
| Pothole |               1,261 |
| Crack   |               2,519 |
| Manhole |                 957 |

The dataset is originally organised for **object detection**. In this project, the polygon annotations are used to extract individual regions so that the problem can be formulated as an image-classification task.

---

## Methodology

The complete pipeline is:

```text
YOLO Polygon Annotations
          ↓
Annotation Parsing
          ↓
Polygon → Bounding Box
          ↓
Damage Crop Extraction
          ↓
Image-Level Train / Validation / Test Split
          ↓
Image Preprocessing + Augmentation
          ↓
MobileNetV2 Transfer Learning
          ↓
Fine-Tuning
          ↓
Model Evaluation
          ↓
Grad-CAM Explanation
          ↓
Prediction + Confidence + Heatmap
```

### 1. Parsing the annotations

Each YOLO polygon annotation contains:

```text
class_id x1 y1 x2 y2 ...
```

The normalised polygon coordinates are converted into pixel coordinates and associated with their corresponding image and class.

This produces a structured annotation table containing the image path, label path, class information, and polygon coordinates.

---

### 2. Extracting individual damage crops

For every annotated object, the polygon is converted into a bounding box.

A small amount of padding is added around the bounding box to preserve some surrounding road context. The resulting region is saved as an individual image crop.

The final crop distribution is:

```text
Pothole    → 1,261 crops
Crack      → 2,518 crops
Manhole    →   957 crops
```

One annotated object did not produce a usable crop, resulting in 4,736 valid crops.

---

### 3. Image-level data splitting

A particularly important part of the pipeline is the way the dataset is split.

The split is performed using the **original source images**, rather than randomly splitting the extracted crops.

This prevents different crops from the same original image from being distributed across training and testing sets, which could otherwise make evaluation overly optimistic.

```text
70% → Training
15% → Validation
15% → Testing
```

---

### 4. Image preprocessing and augmentation

All images are resized to:

```text
224 × 224 × 3
```

MobileNetV2 preprocessing is then applied.

During training, the following augmentations are used:

* Random horizontal flip
* Random rotation
* Random zoom
* Random contrast adjustment

The purpose is to make the classifier less dependent on the exact orientation and appearance of the training images.

---

### 5. Transfer learning with MobileNetV2

The model uses **MobileNetV2 pretrained on ImageNet** as its visual feature extractor.

The classification architecture is:

```text
Input Image
     ↓
Data Augmentation
     ↓
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dropout (0.30)
     ↓
Dense Layer
     ↓
Softmax
     ↓
3 Classes
```

The model contains approximately **2.26 million parameters**, making it considerably smaller than many modern convolutional architectures.

Initially, the MobileNetV2 backbone is frozen and only the classification head is trained.

---

### 6. Fine-tuning

After training the classification head, the upper portion of MobileNetV2 is unfrozen.

The first 100 backbone layers remain frozen while later layers are fine-tuned using a smaller learning rate.

The training process uses:

* **Optimizer:** Adam
* **Initial learning rate:** `1e-3`
* **Fine-tuning learning rate:** `1e-5`
* **Loss:** Sparse Categorical Cross-Entropy
* **Batch size:** 32
* **Early stopping**
* **Learning-rate reduction on validation-loss plateau**

This two-stage strategy allows the model to first learn the new classification task and then gradually adapt higher-level pretrained features to road-damage imagery.

---

### 7. Model evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* Macro F1-score
* Confusion matrix
* Normalised confusion matrix

Macro-averaged metrics are included because the classes are not evenly distributed.

---

### 8. Explainability with Grad-CAM

To understand the model's visual reasoning, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is applied to the final convolutional layer of MobileNetV2.

The Grad-CAM process produces a heatmap indicating the image regions that have the strongest positive influence on the selected class prediction.

The final prediction pipeline therefore produces:

```text
Input Image
     ↓
Predicted Class
     +
Confidence Score
     +
Grad-CAM Heatmap
```

This makes it possible to inspect both the prediction and the visual evidence used by the model.

---

## Results

The final model was evaluated on **692 unseen test crops**.

### Overall performance

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **69.51%** |
| Macro Precision | **63.85%** |
| Macro Recall    | **62.91%** |
| Macro F1        | **63.33%** |

The test set contained:

```text
481 correct predictions
211 incorrect predictions
```

### Per-class performance

| Class   |  Precision |     Recall |   F1-score | Support |
| ------- | ---------: | ---------: | ---------: | ------: |
| Pothole |     58.08% |     54.49% |     56.23% |     178 |
| Crack   | **79.15%** | **83.11%** | **81.08%** |     379 |
| Manhole |     54.33% |     51.11% |     52.67% |     135 |

---

## Findings

The experiments provide several useful observations.

### 1. Crack is the easiest class to recognise

The model performs substantially better on **Crack** images, reaching an F1-score of **81.08%**.

This indicates that the visual characteristics of cracks are more consistently captured by the learned features than those of potholes and manholes.

### 2. Pothole and Manhole classification remains challenging

The F1-scores for Pothole and Manhole are **56.23%** and **52.67%**, respectively.

This shows that overall accuracy alone does not fully describe the model's behaviour. The relatively lower performance of these classes is reflected in the macro F1-score of **63.33%**.

### 3. The dataset is class-imbalanced

Crack instances are considerably more common than Pothole and Manhole instances.

This imbalance is important when interpreting the results because strong performance on Crack contributes significantly to the overall accuracy.

### 4. Transfer learning provides a practical baseline

MobileNetV2 offers a relatively compact architecture while still achieving close to 70% test accuracy on this three-class problem.

The experiment demonstrates that pretrained visual features can be adapted to a specialised road-condition classification task without training a large CNN from scratch.

### 5. Grad-CAM adds an interpretability layer

Grad-CAM provides a visual explanation of the model's prediction by highlighting the regions that influenced the selected class.

The notebook demonstrates these explanations on individual test examples and includes both correct and incorrect predictions for visual inspection.

The explanations are used qualitatively in this project; no separate quantitative evaluation of Grad-CAM faithfulness is performed.

### 6. The project is a classification system, not a complete detector

An important scope distinction is that this project operates on **cropped damage regions**.

It does not yet take a completely new road scene and simultaneously locate and classify every damage instance. That would require an additional object-detection or segmentation stage.

---

## Explainability Example

For a given input crop, the system produces:

```text
Original Image
      ↓
MobileNetV2
      ↓
Prediction: Manhole
Confidence: 67.04%
      ↓
Grad-CAM Heatmap
      ↓
Visual indication of influential image regions
```

The confidence value above is from one demonstration image in the notebook and should not be interpreted as the overall confidence of the model.

---

## Project Structure

The GitHub/Google Drive project contains only the main documentation and notebook:

```text
road-damage-xai/
│
├── .gitignore
|
├── RoadDamage_XAI_Report.pdf
|
├── README.md
│
└── RoadDamage_XAI.ipynb
```

The trained model and experiment result files are generated and stored in the **Kaggle notebook environment** rather than inside the GitHub/Google Drive project.

### Kaggle output structure

During notebook execution, the following files are saved under:

```text
/kaggle/working/
│
├── road_damage_mobilenetv2.keras
├── road_damage_metrics.csv
└── test_predictions.csv
```

Therefore:

* `road_damage_mobilenetv2.keras` → trained MobileNetV2 model
* `road_damage_metrics.csv` → evaluation metrics
* `test_predictions.csv` → predictions generated for the test set

The notebook is responsible for generating these outputs during the experiment.

The original dataset and intermediate crop files are also handled within the Kaggle execution environment and are not stored in the GitHub repository.

---

## Technologies Used

* **Python**
* **TensorFlow / Keras**
* **MobileNetV2**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Grad-CAM**
* **YOLO polygon annotations**

---

## Reproducibility

The notebook uses a fixed random seed:

```python
SEED = 42
```

This seed is applied to Python, NumPy, and TensorFlow operations used in the experiment.

The notebook was executed using **TensorFlow 2.20.0**.

The trained model and evaluation CSV files are generated in the Kaggle notebook environment under:

```text
/kaggle/working/
```

Running the notebook from top to bottom reproduces the data-processing, training, evaluation, and Grad-CAM workflow. The final trained model and result files are produced by the notebook during execution.

---

## Future Work

The current project provides a strong baseline, but several improvements could make the system more useful in real-world road monitoring.

### 1. Move from classification to end-to-end detection

The next step would be to combine localisation and classification so that a complete road image can be processed directly.

For example:

```text
Road Image
    ↓
Object Detector
    ↓
Damage Locations
    ↓
Damage Classes
    ↓
Explainable Predictions
```

This would make the system closer to a practical road-inspection application.

### 2. Improve performance on minority classes

More balanced training data, class-weighting strategies, targeted augmentation, and additional road-damage samples could be investigated to improve Pothole and Manhole recognition.

### 3. Compare alternative CNN architectures

The MobileNetV2 baseline could be compared with architectures such as EfficientNet, ResNet, or newer lightweight vision models to study the trade-off between accuracy, model size, and inference cost.

### 4. Investigate stronger explainability methods

Grad-CAM could be compared with other XAI techniques and evaluated quantitatively to determine whether highlighted regions are genuinely aligned with the annotated damage areas.

### 5. Introduce segmentation

Since the original data contains polygon annotations, a future version could preserve the segmentation information rather than reducing each object to a bounding box.

This could provide more precise localisation of irregular damage regions such as cracks.

### 6. Evaluate real-world robustness

Future experiments should test the model under different lighting conditions, camera viewpoints, weather conditions, road surfaces, and image-quality levels.

This would help measure how well the approach generalises outside the original dataset.

---

## Conclusion

RoadDamage-XAI demonstrates an end-to-end workflow for building an **explainable computer vision classifier for road-condition analysis**.

The project converts YOLO polygon annotations into individual damage crops, uses an image-level data split to reduce leakage, applies MobileNetV2 transfer learning and fine-tuning, evaluates performance using both overall and class-wise metrics, and uses Grad-CAM to make model predictions easier to interpret.

The final model achieves **69.51% test accuracy** and a **63.33% macro F1-score** across Pothole, Crack, and Manhole classes. The results also show an important difference between overall performance and class-level performance, with Crack substantially easier to classify than the other two categories.

Most importantly, the project treats explainability as part of the machine-learning workflow rather than as an afterthought. The combination of **data preparation, transfer learning, quantitative evaluation, error analysis, and visual explanation** provides a practical foundation for future work towards more robust and fully automated road-damage detection systems.

---

## Dataset Source

The project uses:

**Road Damage Dataset: Potholes, Cracks and Manholes**

Dataset identifier:

`lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`

The dataset is used in this project for research and educational experimentation. Please refer to the original dataset page for its licensing and attribution requirements.

---

## Project Scope

This repository is intended as a machine-learning and computer-vision project demonstrating:

```text
Data Preparation
      +
Computer Vision
      +
Transfer Learning
      +
Model Evaluation
      +
Error Analysis
      +
Explainable AI
```

The emphasis is on building a complete and understandable experimental pipeline rather than claiming production-level road-damage detection accuracy.
