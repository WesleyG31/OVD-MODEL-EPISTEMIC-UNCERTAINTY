RQ3 — Semantic–Spatial Confidence Fusion

How does fusing classification confidence with spatial localization quality improve the reliability, ranking, and calibration of open-vocabulary detections in safety-critical driving scenarios?

Figures & Tables

Figure 3.1 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq3\output\figure_3_1_confidence_iou_correlation.png

Figure 3.2 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq3\output\figure_3_2_ranking_stability.png

Figure 3.3 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq3\output\figure_3_3_laece_calibration_bins.png

Table 3.1 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq3\output\table_3_1_correlation_metrics.csv

Table 3.2 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq3\output\table_3_2_localization_calibration.csv


Figure 3.1 — Relationship Between Confidence and Localization Accuracy
Correlation between predicted confidence and ground-truth IoU under different confidence modeling strategies. Score–IoU fusion aligns semantic confidence with spatial accuracy.

Table 3.1 — Confidence–Localization Correlation Metrics
Pearson and Spearman correlation coefficients between predicted confidence and IoU, showing substantial improvement through semantic–spatial fusion.

Figure 3.2 — Detection Ranking Stability Analysis
Comparison of confidence-induced ranking against IoU-based ranking. Score–IoU fusion significantly improves ranking stability.

Table 3.2 — Localization-Aware Calibration Performance
Impact of confidence–localization fusion on localization-aware calibration error and strict detection accuracy.

Figure 3.3 — Localization-Aware Calibration Across Confidence Bins
Analysis of confidence–IoU alignment across bins, demonstrating reduced overconfidence through semantic–spatial fusion.