# Court Detection Evaluation

This document details the evaluation results for the **tennis court keypoint detection** module. The court detector identifies 14 structural keypoints (corners and line intersections) to define the court's geometry in any given frame.

## 1. Dataset & Methodology

* **Dataset:** Evaluated on the validation set from [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector/tree/main).
* **Sample Size:** 2,211 images
* **Total Keypoints Evaluated:** 30,954 (14 keypoints per image)
* **Evaluation Metric:** A predicted keypoint is considered a **True Positive (TP)** if its Euclidean distance from the ground truth is strictly less than **7 pixels**. Otherwise, it is classified as a False Positive (FP) or False Negative (FN).

The evaluation was performed in two stages to measure the impact of the **Homography Matrix**:
1.  **Before Homography (Raw Neural Network):** Evaluates the pure pixel-coordinate predictions made by the TrackNet architecture before any geometric constraints are applied.
2.  **After Homography (Geometric Post-Processing):** Evaluates the points after calculating a 3x3 transformation matrix and projecting a mathematically perfect "reference court" onto the frame.

---

## 2. Quantitative Results

The application of the homography matrix yields a strict improvement across all metrics. By enforcing straight, parallel lines, the homography matrix effectively "corrects" minor deviations from the raw network predictions.

| Metric | Before Homography | After Homography | Delta |
| :--- | :--- | :--- | :--- |
| **True Positives** | 28,874 | 29,663 | *+ 789* |
| **False Positives** | 1,867 | 1,141 | *- 726* |
| **False Negatives** | 111 | 51 | *- 60* |
| **Precision** | 93.93% | **96.30%** | *+ 2.37%* |
| **Accuracy** | 93.61% | **96.15%** | *+ 2.54%* |
| **Mean Pixel Error** | 3.24 px | **2.79 px** | *- 0.45 px* |
| **Median Pixel Error**| 2.24 px | **1.84 px** | *- 0.40 px* |

---

## 3. Analysis

### The Role of Homography
The results validate the necessity of the homography step in the pipeline:
* **Error Reduction:** The mean pixel error dropped from 3.24 to 2.79 pixels, and the median dropped to an impressive 1.84 pixels. 
* **Robustness:** False positives were reduced by nearly 40% (1,867 down to 1,141). When the neural network hallucinates a keypoint slightly off the line due to shadows or player occlusion, the homography matrix forces that point back into structural alignment with the other 13 high-confidence points.

### Conclusion for the Score Pipeline
Achieving a **96.15% accuracy** with a sub-2-pixel median error after homography ensures that the resulting `court_polygon` passed to the scoring logic is highly reliable. This precision is critical for accurately resolving close "in or out" ball-bounce calls near the baselines and sidelines.