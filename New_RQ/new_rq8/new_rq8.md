RQ8 — Joint semantic–geometric calibration for reliability

RQ8: How can semantic confidence and localization quality be jointly calibrated to yield meaningful scores for ranking/selection?

Figures & Tables

Figure 8.1 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq8\output\Fig_RQ8_1_score_iou_reliability.png

Figure 8.2 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq8\output\Fig_RQ8_2_precision_at_k.png

Table 8.1 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq8\output\table_rq8_1_score_iou_alignment.csv

Table 8.2 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq8\output\table_rq8_2_ranking_utility.csv

Figure 8.1 — Reliability of detection scores with respect to geometric quality (mean IoU per confidence bin). Joint calibration substantially improves monotonic alignment between score and localization accuracy.

Table 8.1 — Correlation between detection score and IoU before and after joint calibration.

Figure 8.2 — Precision@K for ranking detections under raw and calibrated scores (log-scale K). Calibration improves ranking quality, supporting reliability-aware selection beyond mAP.

Table 8.2 — Ranking and selection improvements induced by joint calibration at fixed proposal budget.

---