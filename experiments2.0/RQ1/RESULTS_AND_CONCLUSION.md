# RQ1 Results and Conclusion / Resultados y conclusión de RQ1

## English

### Research question

> How can epistemic uncertainty extracted from multiple internal representations of a transformer-based open-vocabulary detector be fused into a reliable uncertainty signal for risk-aware ADAS perception?

### Experimental scope

RQ1 was evaluated with the pinned GroundingDINO Swin-T checkpoint and ten fixed BDD100K category prompts. The confirmatory evaluation partition contained 1,992 images. At the prespecified operational score threshold of 0.20, 47,805 detections were evaluated. Statistical uncertainty was estimated with paired image/sequence-clustered bootstrap resampling, and the two superiority hypotheses were adjusted with the Holm procedure at a familywise alpha of 0.05.

The underlying detector achieved a COCO mAP@[0.50:0.95] of 0.2121, AP50 of 0.3693, AP75 of 0.2038, and AR@100 of 0.4548 on the frozen evaluation partition. These values define the detector-quality context in which the uncertainty results must be interpreted.

### Confirmatory result

The prespecified primary method, `all_internal`, combined semantic, geometric, representation, decoder-reference, score-variation, and stochastic-presence information without using the detector confidence as an input. It did not satisfy the frozen RQ1 success rule.

| Metric | `all_internal` | Confidence baseline | Paired improvement | 95% bootstrap interval for improvement | Holm-adjusted p-value | Decision |
|---|---:|---:|---:|---:|---:|---|
| AUROC | 0.7740 | 0.7737 | +0.0003 | [-0.0051, 0.0054] | 0.8896 | No superiority |
| AURC | 0.3604 | 0.3487 | -0.0118 | [-0.0153, -0.0082] | 1.0000 | Worse than confidence |
| Brier score | 0.1907 | 0.1880 | -0.0027 | [-0.0048, -0.0005] | Not applicable | Non-inferiority passed at margin 0.02 |

Although calibration non-inferiority was satisfied, the method did not improve AUROC and produced a significantly less favorable AURC point estimate and interval. Consequently, the confirmatory result is `not_supported`, and `rq1_answer_supported` is `false`.

### Secondary findings

The secondary nonlinear internal-only model, `all_internal_rf`, produced the strongest overall performance: AUROC 0.8342 (95% CI 0.8299–0.8384), AUPRC 0.8548, AURC 0.3041 (95% CI 0.2982–0.3103), and Brier score 0.1619. The confidence-augmented fusion, `all_plus_confidence`, also outperformed the confidence baseline descriptively, with AUROC 0.8249, AUPRC 0.8480, AURC 0.3184, and Brier score 0.1678.

These results indicate that the internal representations contain complementary information, but that their relationship with detection error is not adequately captured by the prespecified primary linear fusion. Because neither `all_internal_rf` nor `all_plus_confidence` defined the confirmatory success decision, these findings cannot replace or overturn the negative primary result. They should be reported as secondary evidence and independently validated before supporting a general claim.

The MC-pass sensitivity analysis showed a consistent improvement as more stochastic samples were used:

| MC passes | AUROC | AUPRC | AURC |
|---:|---:|---:|---:|
| 2 | 0.7220 | 0.7553 | 0.3981 |
| 5 | 0.7543 | 0.7867 | 0.3745 |
| 10 | 0.7740 | 0.8041 | 0.3604 |

Thus, two or five passes did not reproduce the quality of the ten-pass primary configuration. The added passes improved ranking quality, but they also increased inference cost substantially.

Validation-only robustness analyses showed mixed behavior. Alternative MC seeds retained similar uncertainty AUROC values but only moderate rank agreement with the canonical signal (Spearman approximately 0.86–0.87). Association-IoU changes were more stable (Spearman approximately 0.92–0.95). The synonym prompt condition substantially changed rankings and reduced detector mAP on the 100-image robustness subset. Under Gaussian blur with sigma 2, mean uncertainty decreased by 0.0228, with a paired 95% interval of [-0.0360, -0.0104], even though detector quality deteriorated. This is an important limitation for interpreting the signal as a general out-of-distribution or corruption alarm.

At image level, the endpoint “has any detection error” was not estimable because its prevalence was 1.0. The false-negative endpoint had a prevalence of 0.9704; `all_internal` achieved an AUROC of only 0.4855 for this endpoint, compared with 0.6668 for confidence. The proposed signal must therefore be described as detection-conditioned and cannot be claimed to solve uncertainty for objects that generated no detection.

### Computational result

For the 1,992-image evaluation partition, the recorded synchronized warm-model path required 9,249.3 seconds in total: 755.1 seconds for deterministic inference, 7,969.1 seconds for ten stochastic passes, and 478.1 seconds for aggregation. Throughput was approximately 0.215 images per second, and peak recorded GPU memory was 2.03 GB on an NVIDIA GeForce RTX 4060 Laptop GPU. These measurements characterize the evaluated implementation and are not evidence of real-time ADAS feasibility.

### Conclusion

RQ1 is answered negatively for its prespecified confirmatory method. Under the frozen GroundingDINO Swin-T, BDD100K, prompt, threshold, and ten-pass MC configuration, linearly fusing epistemic uncertainty from multiple internal representations did not produce a more reliable risk-ranking signal than detector confidence. It matched confidence in AUROC, was slightly worse in Brier score while remaining within the non-inferiority margin, and was clearly worse in AURC.

The secondary results nevertheless provide an important qualified finding: the same internal representations supported substantially better error discrimination when fused nonlinearly. The evidence therefore suggests that useful epistemic information is present across representations, but its interactions are nonlinear and sensitive to the fusion rule, number of MC passes, prompts, and image corruption. A defensible paper conclusion is that multi-representation fusion is promising but not automatically reliable; reliability requires a nonlinear fusion mechanism and confirmation on an untouched external dataset or additional detector/backbone. Until such confirmation is available, the strong random-forest result must remain secondary, and all claims must be restricted to detection-conditioned errors for the evaluated Swin-T checkpoint and fixed BDD100K vocabulary.

---

## Español

### Pregunta de investigación

> ¿Cómo puede fusionarse la incertidumbre epistémica extraída de múltiples representaciones internas de un detector de vocabulario abierto basado en transformers para obtener una señal de incertidumbre fiable para percepción ADAS consciente del riesgo?

### Alcance experimental

RQ1 se evaluó con el checkpoint fijado de GroundingDINO Swin-T y diez prompts correspondientes a categorías de BDD100K. La partición confirmatoria contuvo 1.992 imágenes. Con el umbral operativo preespecificado de 0,20 se evaluaron 47.805 detecciones. La incertidumbre estadística se estimó mediante bootstrap pareado agrupado por imagen/secuencia, y las dos hipótesis de superioridad se ajustaron mediante el procedimiento de Holm con alfa familiar de 0,05.

El detector base obtuvo COCO mAP@[0,50:0,95] de 0,2121, AP50 de 0,3693, AP75 de 0,2038 y AR@100 de 0,4548 en la partición congelada. Estos valores establecen el contexto de calidad del detector dentro del cual deben interpretarse los resultados de incertidumbre.

### Resultado confirmatorio

El método primario preespecificado, `all_internal`, combinó información semántica, geométrica, de representación, referencias del decoder, variación del score y presencia estocástica, sin utilizar la confianza del detector como entrada. El método no satisfizo la regla de éxito congelada para RQ1.

| Métrica | `all_internal` | Baseline de confianza | Mejora pareada | Intervalo bootstrap del 95% para la mejora | p ajustado por Holm | Decisión |
|---|---:|---:|---:|---:|---:|---|
| AUROC | 0,7740 | 0,7737 | +0,0003 | [-0,0051; 0,0054] | 0,8896 | Sin superioridad |
| AURC | 0,3604 | 0,3487 | -0,0118 | [-0,0153; -0,0082] | 1,0000 | Peor que confianza |
| Brier score | 0,1907 | 0,1880 | -0,0027 | [-0,0048; -0,0005] | No aplicable | No-inferioridad aprobada con margen 0,02 |

Aunque se cumplió la no-inferioridad de calibración, el método no mejoró AUROC y produjo una estimación e intervalo AURC menos favorables. Por ello, el resultado confirmatorio es `not_supported` y `rq1_answer_supported` es `false`.

### Hallazgos secundarios

El modelo no lineal interno `all_internal_rf` obtuvo el mejor rendimiento general: AUROC 0,8342 (IC 95% 0,8299–0,8384), AUPRC 0,8548, AURC 0,3041 (IC 95% 0,2982–0,3103) y Brier score 0,1619. La fusión complementada con confianza, `all_plus_confidence`, también superó descriptivamente al baseline de confianza, con AUROC 0,8249, AUPRC 0,8480, AURC 0,3184 y Brier score 0,1678.

Estos resultados indican que las representaciones internas contienen información complementaria, pero que su relación con el error de detección no queda adecuadamente capturada por la fusión lineal primaria. Debido a que ni `all_internal_rf` ni `all_plus_confidence` definían la decisión confirmatoria, estos hallazgos no pueden sustituir ni revertir el resultado primario negativo. Deben presentarse como evidencia secundaria y validarse de manera independiente antes de respaldar una afirmación general.

El análisis de sensibilidad al número de pasadas MC mostró una mejora consistente al incrementar las muestras estocásticas:

| Pasadas MC | AUROC | AUPRC | AURC |
|---:|---:|---:|---:|
| 2 | 0,7220 | 0,7553 | 0,3981 |
| 5 | 0,7543 | 0,7867 | 0,3745 |
| 10 | 0,7740 | 0,8041 | 0,3604 |

Por tanto, dos o cinco pasadas no reprodujeron la calidad de la configuración primaria de diez pasadas. Las pasadas adicionales mejoraron el ordenamiento del riesgo, pero aumentaron considerablemente el costo de inferencia.

Los análisis de robustez realizados únicamente sobre validación mostraron un comportamiento mixto. Semillas MC alternativas conservaron valores semejantes de AUROC de incertidumbre, pero sólo una concordancia moderada de ranking con la señal canónica (Spearman aproximadamente 0,86–0,87). Los cambios en el IoU de asociación fueron más estables (Spearman aproximadamente 0,92–0,95). La condición con prompts sinónimos modificó sustancialmente el ranking y redujo el mAP del detector en el subconjunto de robustez de 100 imágenes. Con desenfoque gaussiano sigma 2, la incertidumbre media disminuyó en 0,0228, con un intervalo pareado del 95% de [-0,0360; -0,0104], aunque la calidad del detector se deterioró. Esto limita la interpretación de la señal como una alarma general ante corrupciones o datos fuera de distribución.

A nivel de imagen, el endpoint “existe algún error de detección” no pudo estimarse porque su prevalencia fue 1,0. El endpoint de falsos negativos tuvo prevalencia 0,9704; `all_internal` alcanzó un AUROC de sólo 0,4855, frente a 0,6668 para confianza. En consecuencia, la señal propuesta debe describirse como condicionada a las detecciones y no puede afirmarse que resuelva la incertidumbre de objetos que no produjeron ninguna detección.

### Resultado computacional

Para las 1.992 imágenes de evaluación, la ruta sincronizada con el modelo precargado requirió 9.249,3 segundos: 755,1 segundos de inferencia determinista, 7.969,1 segundos para diez pasadas estocásticas y 478,1 segundos de agregación. El throughput fue aproximadamente 0,215 imágenes por segundo y el pico de memoria GPU registrado fue 2,03 GB en una NVIDIA GeForce RTX 4060 Laptop GPU. Estas mediciones caracterizan la implementación evaluada y no constituyen evidencia de viabilidad ADAS en tiempo real.

### Conclusión

RQ1 se responde negativamente para su método confirmatorio preespecificado. Bajo la configuración congelada de GroundingDINO Swin-T, BDD100K, prompts, umbral y diez pasadas MC, la fusión lineal de incertidumbre epistémica procedente de múltiples representaciones internas no produjo una señal de ordenamiento del riesgo más fiable que la confianza del detector. Igualó aproximadamente a la confianza en AUROC, fue ligeramente peor en Brier aunque permaneció dentro del margen de no-inferioridad, y fue claramente peor en AURC.

Los resultados secundarios aportan, no obstante, un hallazgo cualificado importante: las mismas representaciones internas permitieron una discriminación de errores sustancialmente mejor al fusionarse de manera no lineal. La evidencia sugiere que existe información epistémica útil distribuida entre las representaciones, pero sus interacciones son no lineales y sensibles a la regla de fusión, el número de pasadas MC, los prompts y las corrupciones de imagen. Una conclusión defendible para el artículo es que la fusión multirrepresentación es prometedora, pero no automáticamente fiable: requiere un mecanismo de fusión no lineal y confirmación en un conjunto externo intacto o con un detector/backbone adicional. Hasta disponer de esa confirmación, el resultado fuerte del random forest debe conservarse como secundario y las conclusiones deben limitarse a errores condicionados a detecciones para el checkpoint Swin-T y el vocabulario fijo de BDD100K evaluados.
