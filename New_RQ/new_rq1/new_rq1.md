How can epistemic uncertainty extracted from multiple internal representations of a transformer-based open-vocabulary detector be fused into a reliable uncertainty signal for risk-aware ADAS perception?

Figures & Tables to Use

Figure 1.1 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq1\output\figure_1_1_decoder_uncertainty.png

Figure 1.2 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq1\output\figure_1_2_reliability_diagrams.png

Figure 1.3 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq1\output\figure_1_3_fusion_strategies.png

Table 1.1 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq1\output\table_1_1_layer_calibration.csv

Table 1.2 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq1\output\table_1_2_fusion_impact.csv


Table 1.1 — Calibration Performance Across Decoder Representations
Comparison of calibration metrics obtained from uncertainty estimates extracted independently from individual decoder layers. Results indicate that no single representation yields optimal calibration, motivating representation-level uncertainty fusion.

Figure 1.2 — Reliability Diagrams for Single-Layer and Fused Uncertainty Estimates
Reliability diagrams comparing individual decoder-layer uncertainty estimates with representation-level fusion strategies. Fusion alters calibration behavior, revealing trade-offs between alignment and discriminative uncertainty quality.

Table 1.2 — Impact of Representation-Level Uncertainty Fusion on Reliability and Risk
Effect of different representation-level uncertainty fusion strategies on calibration (ECE), selective prediction performance (AURC), and detection accuracy. Variance-based fusion yields superior risk behavior while preserving detection performance.

Figure 1.3 — Discriminative Power of Representation-Level Fusion Strategies
Separation of true and false detections under different uncertainty fusion strategies. Variance-based fusion improves uncertainty discrimination, enhancing its suitability for risk-aware decision making.


#############################################################

