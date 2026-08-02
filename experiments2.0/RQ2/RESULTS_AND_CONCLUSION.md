# RQ2: Confirmatory Results and Conclusion / Resultados confirmatorios y conclusión

- [English](#english)
- [Español](#español)

## English

### Research question

> How does fusing deterministic and stochastic uncertainty estimators improve
> reliability in open-vocabulary object detection?

### Scope of the evidence

The results come from the frozen confirmatory evaluation of GroundingDINO
Swin-T on BDD100K. The experiment used 5,600 training images, 2,400 validation
images, and 1,992 confirmatory test images, with no sequence overlap between
partitions. The primary evaluation included 47,805 detections with scores at
or above 0.20; the error prevalence was 0.5685.

Reliability is defined here as the ability to rank incorrect detections by
uncertainty, reduce risk by rejecting uncertain detections, and produce
calibrated error probabilities. Fusion does not change the detector's
predictions and therefore does not imply an improvement in mAP.

### Primary results

The primary comparison evaluated learned logistic fusion against two
estimators from the same model family: one using only deterministic features
and one using only stochastic features.

| Method | AUROC ↑ | AUPRC ↑ | AURC ↓ | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Learned deterministic | 0.6224 | 0.6952 | 0.4890 | 0.2329 | 0.6568 | 0.0045 |
| Learned stochastic | 0.7562 | 0.7838 | 0.3676 | 0.1966 | 0.5783 | 0.0085 |
| **Learned fusion** | **0.7755** | **0.8067** | **0.3592** | **0.1902** | **0.5640** | 0.0106 |

All four primary tests favored fusion after Holm correction. For AURC, a
positive improvement denotes a reduction in the area under the risk-coverage
curve.

| Comparison | AUROC improvement [95% CI] | AURC reduction [95% CI] | Holm-adjusted p |
|---|---:|---:|---:|
| Fusion vs. deterministic | 0.1531 [0.1475, 0.1588] | 0.1298 [0.1244, 0.1353] | 0.001999 |
| Fusion vs. stochastic | 0.0193 [0.0169, 0.0216] | 0.0084 [0.0065, 0.0103] | 0.001999 |

The prespecified confirmatory criterion was therefore met:

```text
primary_inference.success_criterion_met = true
```

The gain over the deterministic estimator was large. The gain over the
stochastic estimator was smaller, but consistent, with wholly favorable
confidence intervals and significance after controlling the four primary
comparisons.

### Estimator complementarity

On validation data, the Spearman correlation between deterministic and
stochastic scores was 0.2675. Moreover, 40.64% of detections fell into opposite
quadrants relative to the medians of the two estimators. Together with the
confirmatory improvement from fusion, these results indicate that the two
families contain partially complementary information rather than duplicate
versions of the same signal.

### Comparison with detector confidence

Primary fusion without confidence was not uniformly better than the ordinary
`1 - confidence` control:

| Method | AUROC ↑ | AURC ↓ | Brier ↓ |
|---|---:|---:|---:|
| Confidence | 0.7737 | **0.3487** | **0.1880** |
| Learned fusion | 0.7755 | 0.3592 | 0.1902 |
| Fusion + confidence | **0.8262** | 0.3174 | **0.1672** |
| Nonlinear fusion | 0.8187 | **0.3138** | 0.1689 |

The AUROC difference between primary fusion and confidence was small (0.0018;
95% CI [-0.0033, 0.0071]), while confidence achieved a lower AURC than primary
fusion. In contrast, explicitly retaining confidence in the fusion or allowing
nonlinear interactions produced clear secondary improvements. These variants
were prespecified sensitivity analyses and do not retrospectively replace the
primary method.

Fixed equal-weight fusion was also insufficient: its AUROC was 0.6805 and its
AURC was 0.4195. The benefit therefore does not arise from combining signals
indiscriminately, but from learning an appropriate relationship between them.

### Calibration, number of samples, and robustness

Primary fusion improved Brier score and NLL over both standalone components,
but its ECE was higher. The calibration evidence is therefore favorable under
proper scoring rules, but not uniform across every calibration metric.

The MC-pass analysis showed progressive improvement:

| MC passes | AUROC ↑ | AURC ↓ | Brier ↓ |
|---:|---:|---:|---:|
| 2 | 0.7271 | 0.3956 | 0.2080 |
| 5 | 0.7574 | 0.3726 | 0.1976 |
| 10 | 0.7755 | 0.3592 | 0.1902 |

Fusion outperformed both standalone components in AUROC and AURC at all four
evaluated score thresholds: 0.05, 0.10, 0.20, and 0.30. Subgroup robustness was
more heterogeneous. Fusion outperformed the deterministic estimator in every
eligible subgroup, but simultaneously outperformed the stochastic estimator
in both AUROC and AURC for 2 of 10 categories, 2 of 3 object sizes, every scene
and time-of-day group, and 5 of 6 weather groups. Subgroup results are
descriptive and do not alter the aggregate primary conclusion.

### Computational cost

Across the 1,992 confirmatory images, the deterministic pass accumulated 755.1
seconds and the ten stochastic passes accumulated 7,969.1 seconds. The
incremental stochastic cost was approximately 10.55 times the deterministic
cost, and the complete fused path processed 0.192 images per second. Sharing
inference across the research questions avoids repeating this computation in
the study, but an operational implementation of fusion would still require
the stochastic samples.

For reference, the frozen detector achieved mAP@[0.50:0.95] = 0.2121,
AP50 = 0.3693, and AP75 = 0.2038. These metrics characterize the base detector;
fusion evaluates the reliability of its detections without changing those
predictions.

### Conclusion

The confirmatory results support the conclusion that fusing deterministic and
stochastic uncertainty signals improves open-vocabulary object-detection
reliability relative to using either family alone. The improvement is
particularly large over the deterministic estimator and more modest, although
statistically robust, over the stochastic estimator. This demonstrates that
deterministic decoder dynamics contribute information that complements the
variation induced through DropPath.

The conclusion should not, however, be stated as claiming that every fusion is
better than every baseline. Fixed fusion was weak, and primary linear fusion
did not outperform ordinary confidence in selective risk. The strongest
secondary results appeared when confidence was retained as an additional input
or when nonlinear relationships were modeled. The evidence therefore supports
**learned, structured fusion of complementary signals**, rather than naive
averaging or a universal improvement of the detector.

The conclusion is limited to detections produced by GroundingDINO Swin-T, the
vocabulary used in this experiment, and the confirmatory BDD100K partition. It
does not cover completely missed objects, demonstrate an improvement in mAP,
or provide a general ADAS safety guarantee.

### Traceability

- Complete results: `RQ2/outputs/metrics.json`.
- Primary comparisons: `RQ2/outputs/Table_RQ2_primary_inference.csv`.
- Main metrics: `RQ2/outputs/Table_RQ2_main.csv`.
- MC sensitivity: `RQ2/outputs/Table_RQ2_mc_pass_sensitivity.csv`.
- Curves and calibration: `RQ2/outputs/Fig_RQ2_*.png` and
  `RQ2/outputs/Fig_RQ2_*.pdf`.
- Integrity manifest: `RQ2/outputs/report_manifest.json`.

The hashes of the core artifacts and all 15 report artifacts were verified
after execution. All required results were finite, and the artifacts remain
compatible with the current code and configuration.

---

## Español

### Pregunta de investigación

> ¿Cómo mejora la fiabilidad en detección de objetos de vocabulario abierto la
> fusión de estimadores de incertidumbre deterministas y estocásticos?

### Alcance de la evidencia

Los resultados corresponden a la evaluación confirmatoria congelada de
GroundingDINO Swin-T sobre BDD100K. Se utilizaron 5.600 imágenes para
entrenamiento, 2.400 para validación y 1.992 para la prueba confirmatoria, sin
solapamiento entre secuencias. La evaluación primaria se realizó sobre 47.805
detecciones con puntuación mayor o igual que 0.20; la prevalencia de errores
fue 0.5685.

La fiabilidad se entiende aquí como la capacidad de ordenar detecciones
incorrectas por incertidumbre, reducir el riesgo al rechazar detecciones
inciertas y producir probabilidades de error calibradas. La fusión no modifica
las predicciones del detector y, por tanto, no implica una mejora de mAP.

### Resultados primarios

La comparación principal enfrentó la fusión logística aprendida con dos
estimadores de la misma familia: uno que utiliza solamente las características
deterministas y otro que utiliza solamente las características estocásticas.

| Método | AUROC ↑ | AUPRC ↑ | AURC ↓ | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Determinista aprendido | 0.6224 | 0.6952 | 0.4890 | 0.2329 | 0.6568 | 0.0045 |
| Estocástico aprendido | 0.7562 | 0.7838 | 0.3676 | 0.1966 | 0.5783 | 0.0085 |
| **Fusión aprendida** | **0.7755** | **0.8067** | **0.3592** | **0.1902** | **0.5640** | 0.0106 |

Las cuatro pruebas primarias favorecieron a la fusión después de aplicar la
corrección de Holm. En la columna de AURC, una mejora positiva representa una
reducción del área bajo la curva riesgo-cobertura.

| Comparación | Mejora de AUROC [IC 95%] | Reducción de AURC [IC 95%] | p ajustado de Holm |
|---|---:|---:|---:|
| Fusión frente a determinista | 0.1531 [0.1475, 0.1588] | 0.1298 [0.1244, 0.1353] | 0.001999 |
| Fusión frente a estocástico | 0.0193 [0.0169, 0.0216] | 0.0084 [0.0065, 0.0103] | 0.001999 |

Por consiguiente, se cumplió el criterio confirmatorio preespecificado:

```text
primary_inference.success_criterion_met = true
```

La ganancia frente al estimador determinista fue grande. La ganancia frente al
estocástico fue menor en magnitud, pero consistente, con intervalos de
confianza completamente favorables y significación después del control de las
cuatro comparaciones primarias.

### Complementariedad de los estimadores

En validación, la correlación de Spearman entre las puntuaciones determinista y
estocástica fue 0.2675. Además, el 40.64% de las detecciones se ubicó en
cuadrantes opuestos respecto a las medianas de ambos estimadores. Junto con la
mejora confirmatoria de la fusión, estos resultados indican que las dos
familias contienen información parcialmente complementaria y no son simples
duplicados de la misma señal.

### Comparación con la confianza del detector

La fusión primaria sin confianza no fue uniformemente mejor que el control
ordinario `1 - confidence`:

| Método | AUROC ↑ | AURC ↓ | Brier ↓ |
|---|---:|---:|---:|
| Confianza | 0.7737 | **0.3487** | **0.1880** |
| Fusión aprendida | 0.7755 | 0.3592 | 0.1902 |
| Fusión + confianza | **0.8262** | 0.3174 | **0.1672** |
| Fusión no lineal | 0.8187 | **0.3138** | 0.1689 |

La diferencia de AUROC entre la fusión primaria y confianza fue pequeña
(0.0018; IC 95% [-0.0033, 0.0071]), mientras que confianza obtuvo menor AURC
que la fusión primaria. En cambio, incorporar explícitamente la confianza a la
fusión o permitir interacciones no lineales produjo mejoras secundarias claras.
Estas variantes son análisis preespecificados de sensibilidad y no reemplazan
retrospectivamente el método primario.

La combinación fija mediante un promedio de puntuaciones tampoco fue
suficiente: su AUROC fue 0.6805 y su AURC 0.4195. Esto muestra que el beneficio
no procede de combinar indiscriminadamente las señales, sino de aprender una
relación adecuada entre ellas.

### Calibración, número de muestras y robustez

La fusión primaria mejoró Brier y NLL frente a ambos componentes, pero su ECE
fue mayor. Por ello, la evidencia de calibración es favorable en las reglas de
puntuación propias, pero no uniforme para todas las métricas de calibración.

El análisis del número de muestras MC mostró una mejora progresiva:

| Muestras MC | AUROC ↑ | AURC ↓ | Brier ↓ |
|---:|---:|---:|---:|
| 2 | 0.7271 | 0.3956 | 0.2080 |
| 5 | 0.7574 | 0.3726 | 0.1976 |
| 10 | 0.7755 | 0.3592 | 0.1902 |

La fusión superó a los dos componentes en AUROC y AURC en los cuatro umbrales
de puntuación evaluados: 0.05, 0.10, 0.20 y 0.30. La robustez por subgrupos fue
más heterogénea. La fusión superó al determinista en todos los subgrupos
elegibles, pero frente al estocástico lo hizo simultáneamente en AUROC y AURC
en 2 de 10 categorías, 2 de 3 tamaños de objeto, todos los grupos de escena y
hora del día, y 5 de 6 grupos meteorológicos. Los subgrupos son descriptivos y
no alteran la conclusión primaria agregada.

### Coste computacional

En las 1.992 imágenes confirmatorias, la pasada determinista acumuló 755.1
segundos y las diez pasadas estocásticas 7.969.1 segundos. El coste incremental
estocástico fue aproximadamente 10.55 veces el coste determinista y el camino
fusionado completo procesó 0.192 imágenes por segundo. Compartir las
inferencias entre las preguntas de investigación evita repetir este cálculo en
el estudio, pero una implementación operativa de la fusión seguiría necesitando
las muestras estocásticas.

Como referencia, el detector congelado obtuvo mAP@[0.50:0.95] = 0.2121,
AP50 = 0.3693 y AP75 = 0.2038. Estas métricas caracterizan el detector base;
la fusión evalúa la fiabilidad de sus detecciones sin cambiar esas predicciones.

### Conclusión

Los resultados confirmatorios respaldan que la fusión de señales de
incertidumbre deterministas y estocásticas mejora la fiabilidad de la detección
de objetos de vocabulario abierto frente al uso aislado de cualquiera de las
dos familias. La mejora es especialmente grande respecto al estimador
determinista y más modesta, aunque estadísticamente robusta, respecto al
estocástico. Esto demuestra que las dinámicas deterministas del decodificador
aportan información complementaria a la variación obtenida mediante DropPath.

No obstante, la conclusión no debe formularse como que cualquier fusión es
mejor que cualquier baseline. La combinación fija fue débil y la fusión lineal
primaria no superó a la confianza ordinaria en riesgo selectivo. Los mejores
resultados secundarios aparecieron cuando la confianza se conservó como una
entrada adicional o cuando se modelaron relaciones no lineales. En
consecuencia, la evidencia apoya una **fusión aprendida y estructurada de
señales complementarias**, no un promedio ingenuo ni una mejora universal del
detector.

La conclusión queda limitada a las detecciones producidas por GroundingDINO
Swin-T, el vocabulario utilizado y la partición confirmatoria de BDD100K. No
cubre objetos completamente omitidos, no demuestra una mejora de mAP y no debe
interpretarse como una garantía general de seguridad para ADAS.

### Trazabilidad

- Resultados completos: `RQ2/outputs/metrics.json`.
- Comparaciones primarias: `RQ2/outputs/Table_RQ2_primary_inference.csv`.
- Métricas principales: `RQ2/outputs/Table_RQ2_main.csv`.
- Sensibilidad MC: `RQ2/outputs/Table_RQ2_mc_pass_sensitivity.csv`.
- Curvas y calibración: `RQ2/outputs/Fig_RQ2_*.png` y `RQ2/outputs/Fig_RQ2_*.pdf`.
- Manifiesto de integridad: `RQ2/outputs/report_manifest.json`.

Los hashes de los artefactos centrales y de los 15 elementos del informe fueron
verificados después de la ejecución. Todos los resultados obligatorios fueron
finitos y los artefactos permanecen compatibles con el código y la configuración
actuales.
