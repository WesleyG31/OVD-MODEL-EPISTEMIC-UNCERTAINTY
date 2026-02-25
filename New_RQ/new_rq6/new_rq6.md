RQ6 — Decoder dynamics as epistemic uncertainty signals

RQ6: What intrinsic properties of transformer decoder dynamics encode epistemic uncertainty in OVD, and when does inter-layer variance reliably proxy model uncertainty?

Figures & Tables

Figure 6.1 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Fig_RQ6_1_decoder_variance.png

Figure 6.2 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Fig_RQ6_2_auroc_by_layer.png

Table 6.1 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Table_RQ6_1.csv

Table 6.2 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Table_RQ6_2.csv

Table 6.2a = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Table_RQ6_2a_failure.csv

Table 6.2b = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Table_RQ6_2b_amplified.csv

Figure 6.1 — Inter-layer bounding-box variance across decoder depth for true positives and false positives. Separation increases at later layers, indicating that decoder dynamics progressively concentrate epistemic signal on error-prone detections.

Table 6.1 — Layer-wise diagnostics of decoder-variance uncertainty. Later layers exhibit improved error discrimination and better risk–coverage characteristics.

Figure 6.2 — AUROC of uncertainty-based error detection as a function of decoder layer. Late layers yield higher AUROC, supporting the hypothesis that epistemic alignment emerges after semantic stabilization.

Table 6.2 — Conditions under which inter-layer variance becomes less predictive of epistemic uncertainty.

---