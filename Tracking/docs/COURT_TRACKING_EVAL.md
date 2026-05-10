# Court Detection Evaluation

This document details the evaluation results for the **tennis court keypoint detection** module. The court detector identifies 14 structural keypoints (corners and line intersections) to define the court's geometry in any given frame.

## 1. Dataset & Methodology

* **Dataset:** Evaluated on the validation set from [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector/tree/main).
* **Sample Size:** 2,211 images
* **Total Keypoints Evaluated:** 30,954 (14 keypoints per image)
* **Evaluation Metric:** A predicted keypoint is considered a **True Positive (TP)** if its Euclidean distance from the ground truth is strictly less than **7 pixels**. Otherwise, it is classified as a False Positive (FP) or False Negative (FN).

---

## 2. Quantitative Results

| Metric | Before Homography |
| :--- | :--- |
| **True Positives** | 28,874 |
| **False Positives** | 1,867 |
| **False Negatives** | 111 |
| **Precision** | 93.93% |
| **Accuracy** | 93.61% |
| **Mean Pixel Error** | 3.24 px |
| **Median Pixel Error**| 2.24 px | 

---

## 3. Analysis

### Conclusion for the Score Pipeline
Achieving a **93.61% accuracy** with a sub-3-pixel median error after homography ensures that the resulting `court_polygon` passed to the scoring logic is reliable. This precision is critical for accurately resolving close "in or out" ball-bounce calls near the baselines and sidelines.