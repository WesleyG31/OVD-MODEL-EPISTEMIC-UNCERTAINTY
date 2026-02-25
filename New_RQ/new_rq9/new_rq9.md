RQ9 — Robustness and stability limits under distribution shift

RQ9: Which components degrade first under semantic/sensory shifts, and what does this reveal about post-hoc reliability limits?

Figures & Tables

Figure 9.1 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq9\output\Fig_RQ9_1_shift_degradation.png

Figure 9.2 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq9\output\Fig_RQ9_2_map_vs_shift.png

Table 9.1 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq9\output\Table_RQ9_1_shift_stress_test.csv

Table 9.2 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq9\output\Table_RQ9_2_component_ablation.csv

Figure 9.1 — Metric degradation with increasing shift severity. Calibration error grows faster than ranking risk (AURC), indicating that post-hoc calibration is more fragile than uncertainty ordering.

Table 9.1 — Performance and reliability under controlled shift. Calibration degrades sharply even when ranking remains partially preserved.

Figure 9.2 — Accuracy collapse (mAP) under increasing shift severity. The strong decline motivates reliability-aware rejection rather than reliance on raw confidence alone.

Table 9.2 — Component ablation under shift. Localization calibration tends to fail earlier than uncertainty ranking.

---