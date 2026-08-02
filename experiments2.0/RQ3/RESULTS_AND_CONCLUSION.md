# RQ3 — Results and conclusion / Resultados y conclusión

[Español](#español) | [English](#english)

## Español

## Pregunta de investigación

**How does fusing classification confidence with spatial localization quality improve reliability, ranking, and calibration for pinned GroundingDINO Swin-T detections on frozen BDD100K driving scenes?**

## Alcance de la evidencia

Estos resultados corresponden a la ejecución confirmatoria completa del protocolo congelado el 31 de julio de 2026. El experimento utilizó GroundingDINO Swin-T con el checkpoint, prompts, semillas, particiones y umbral operativo predefinidos. La interpretación se limita a detecciones condicionadas a que el modelo haya producido una propuesta; por tanto, no evalúa objetos omitidos ni demuestra generalización a otros detectores, datasets o dominios.

La partición experimental fue:

| Partición | Imágenes | Detecciones extraídas | Uso |
|---|---:|---:|---|
| Entrenamiento | 5,600 | 1,195,746 | Ajuste de estimadores |
| Validación | 2,400 | 510,987 | Selección y calibración independientes por grupo |
| Prueba confirmatoria | 1,992 | 422,843 | Evaluación final |

El análisis principal se realizó sobre **47,805 detecciones** con confianza de clasificación `score >= 0.20`, según lo establecido antes de observar los resultados de prueba. La prevalencia de error en este universo operativo fue 0.5685.

## Resultados principales

### Fiabilidad, discriminación de errores y calibración

La fusión principal, `product_fusion`, combina multiplicativamente la confianza de clasificación con una estimación aprendida de calidad espacial. Sus resultados se compararon con la confianza sola y con un control no espacial de igual capacidad.

| Método | AUROC ↑ | AUPRC ↑ | AURC ↓ | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Confianza | 0.7737 | 0.7834 | 0.3487 | 0.1880 | 0.5606 | 0.00955 |
| Control no espacial | 0.8046 | 0.8308 | 0.3316 | 0.1767 | 0.5312 | 0.00927 |
| Fusión espacial por producto | **0.8056** | 0.8194 | **0.3261** | **0.1749** | **0.5277** | **0.00850** |

Frente a la confianza sola, la fusión produjo mejoras confirmatorias en las tres métricas primarias:

| Métrica primaria | Mejora de la fusión | IC bootstrap del 95 % | p ajustado de Holm | Decisión |
|---|---:|---:|---:|---|
| AUROC | +0.03190 | [0.02873, 0.03504] | 0.002999 | Mejora significativa |
| AURC | +0.02257 | [0.02069, 0.02439] | 0.002999 | Mejora significativa |
| Brier | +0.01313 | [0.01192, 0.01428] | 0.002999 | Mejora significativa |

Para AURC y Brier, una mejora positiva en la tabla representa la reducción de una métrica en la que valores menores son mejores. En términos relativos, la fusión redujo el AURC aproximadamente un 6.5 % y el Brier aproximadamente un 7.0 % respecto a la confianza.

Frente al control no espacial de igual capacidad, la evidencia fue más limitada:

| Métrica primaria | Mejora de la fusión | IC bootstrap del 95 % | p ajustado de Holm | Decisión |
|---|---:|---:|---:|---|
| AUROC | +0.00105 | [-0.00176, 0.00393] | 0.2409 | No significativa |
| AURC | +0.00553 | [0.00390, 0.00717] | 0.002999 | Mejora significativa |
| Brier | +0.00183 | [0.00079, 0.00287] | 0.002999 | Mejora significativa |

En total, **cinco de los seis contrastes primarios** sobrevivieron la corrección de Holm con `alpha = 0.05`. El criterio confirmatorio global predefinido exigía que los seis contrastes fueran favorables y significativos. Por ello, el resultado correcto es:

```text
nominal_calibration_criterion_met = true
nominal_ranking_criterion_met     = false
success_criterion_met             = false
```

Este resultado no implica ausencia de beneficio. Indica que la fusión añade valor claro respecto a la confianza sola y mejora el riesgo selectivo y el Brier respecto al control de capacidad, pero no demuestra una ventaja adicional en AUROC sobre ese control no espacial.

### Riesgo selectivo

La fusión permitió conservar una mayor proporción de detecciones manteniendo riesgos bajos:

| Restricción de riesgo | Confianza | Control no espacial | Fusión espacial |
|---|---:|---:|---:|
| Cobertura con riesgo <= 0.05 | 0.42 % | 0.52 % | **2.60 %** |
| Cobertura con riesgo <= 0.10 | 2.43 % | 7.42 % | **10.84 %** |
| Cobertura con riesgo <= 0.20 | 22.62 % | 26.12 % | **28.03 %** |

Asimismo, al retener el 50 % de las detecciones menos inciertas, el riesgo disminuyó de 0.3682 con confianza a 0.3434 con la fusión. Esto respalda el uso de la calidad espacial para rechazo selectivo, priorización de revisión o escalamiento a un subsistema más conservador.

### Ranking detector estándar

El beneficio en discriminación de errores no se trasladó a una mejora del ranking COCO de las detecciones. Al reordenar las 422,843 propuestas mediante la fusión, las métricas estándar disminuyeron:

| Método | mAP@[.50:.95] ↑ | AP50 ↑ | AP75 ↑ |
|---|---:|---:|---:|
| Confianza | **0.2121** | **0.3693** | **0.2038** |
| Control no espacial | 0.2094 | 0.3656 | 0.2013 |
| Fusión espacial | 0.2031 | 0.3508 | 0.1988 |

La fusión redujo el mAP en 0.0090 puntos absolutos, aproximadamente un 4.2 % relativo, frente a la confianza. Este resultado no contradice la mejora de AUROC y AURC: las métricas primarias evalúan la capacidad de ordenar el riesgo de error dentro del universo operativo `score >= 0.20`, mientras que COCO AP evalúa el ranking clase-específico y la curva precisión–recobrado sobre todas las propuestas. En consecuencia, no debe afirmarse que la fusión mejora el ranking detector en sentido general.

## Evidencia sobre la calidad espacial

La taxonomía de las 47,805 detecciones operativas fue:

| Resultado | Detecciones | Proporción |
|---|---:|---:|
| Bien localizada y verdadero positivo | 20,627 | 43.15 % |
| Bien localizada pero error de detección | 4,604 | 9.63 % |
| Pobremente localizada | 22,574 | 47.22 % |

Hubo 27,178 errores de detección en total. De ellos, 22,574, aproximadamente **83.1 %**, fueron detecciones pobremente localizadas. Esto confirma que la calidad espacial captura un mecanismo relevante de error que la confianza de clasificación por sí sola no representa adecuadamente. El IoU medio clase-independiente fue 0.4674.

El estimador de calidad espacial alcanzó AUROC 0.7432 para distinguir localización correcta a IoU 0.50 y AUROC 0.7944 al evaluar el criterio más estricto de IoU 0.75. La señal espacial es, por tanto, informativa, aunque insuficiente para sustituir la confianza semántica.

## Robustez y análisis de sensibilidad

### Número de pases Monte Carlo

El comportamiento mejoró gradualmente al aumentar los pases Monte Carlo:

| Pases MC | AUROC ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
|---:|---:|---:|---:|---:|
| 2 | 0.7990 | 0.3318 | 0.1784 | 0.00797 |
| 5 | 0.8031 | 0.3276 | 0.1761 | **0.00751** |
| 10 | **0.8056** | **0.3261** | **0.1749** | 0.00850 |

La tendencia de AUROC, AURC y Brier favorece diez pases, mientras que ECE presenta una variación pequeña y no monótona. Cinco pases ofrecen resultados cercanos a diez, pero el análisis confirmatorio debe conservar los diez pases predefinidos.

### Umbral de confianza

La utilidad de la fusión dependió del régimen operativo:

| Umbral | Filas | AUROC confianza | AUROC fusión | AURC confianza | AURC fusión | Brier confianza | Brier fusión |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 422,843 | **0.9144** | 0.9033 | 0.7943 | **0.7935** | **0.0401** | 0.0435 |
| 0.10 | 152,317 | 0.8586 | **0.8621** | 0.6142 | **0.6073** | 0.0963 | **0.0924** |
| 0.20 | 47,805 | 0.7737 | **0.8056** | 0.3487 | **0.3261** | 0.1880 | **0.1749** |
| 0.30 | 22,421 | 0.7093 | **0.7630** | 0.2090 | **0.1790** | 0.2004 | **0.1849** |

En el umbral 0.05, la fusión empeoró AUROC y Brier, aunque mejoró marginalmente AURC. Por ello, la conclusión principal se restringe al umbral operativo predefinido de 0.20 y no debe extrapolarse a todas las propuestas de baja confianza.

## Escenarios y subgrupos de conducción

De manera descriptiva, sin pruebas confirmatorias independientes por subgrupo, la fusión mejoró simultáneamente AUROC, AURC y Brier frente a la confianza en 18 de 26 subgrupos estimables. Se observaron mejoras especialmente claras en escenas residenciales, estacionamientos, conducción diurna, clima nublado y objetos pequeños.

Sin embargo, el efecto no fue uniforme entre categorías:

| Categoría | AUROC confianza | AUROC fusión | Cambio |
|---|---:|---:|---:|
| Automóvil | 0.7977 | **0.8225** | +0.0248 |
| Persona | 0.8122 | **0.8164** | +0.0042 |
| Señal de tráfico | 0.7804 | **0.7959** | +0.0155 |
| Bicicleta | **0.8624** | 0.8292 | -0.0332 |
| Bus | **0.9230** | 0.8897 | -0.0333 |
| Motocicleta | **0.8691** | 0.8198 | -0.0493 |
| Rider | **0.5420** | 0.5085 | -0.0335 |
| Camión | **0.8276** | 0.7996 | -0.0280 |

El deterioro en bicicleta, motocicleta y rider es especialmente relevante para seguridad vial. Estos resultados impiden recomendar una política universal de sustitución de la confianza por la fusión sin salvaguardas específicas por categoría y validación adicional para usuarios vulnerables de la vía.

## Integridad y reproducibilidad

La auditoría posterior a la ejecución confirmó:

- 9,992 de 9,992 pares de shards de características e imágenes con hashes válidos.
- Reutilización de los 9,992 shards compartidos y cero inferencias GPU duplicadas durante las solicitudes de RQ3.
- Diez de diez artefactos de modelos con SHA-256 válido.
- Dieciocho de dieciocho artefactos del informe con SHA-256 válido.
- Cero grupos compartidos entre selección y calibración.
- Cero solapamiento entre las particiones científicas de entrenamiento, validación y prueba.
- 2,000 réplicas bootstrap por análisis y 96,000 registros bootstrap en total.
- Cuarenta y nueve pruebas automatizadas superadas.
- Todas las métricas puntuales requeridas finitas.

No se detectaron señales de ejecución parcial, corrupción, contaminación entre particiones o reutilización de modelos de RQ1/RQ2.

## Conclusión y respuesta a la RQ

En GroundingDINO Swin-T sobre las particiones congeladas de BDD100K, fusionar la confianza de clasificación con una estimación de calidad espacial **mejora de forma significativa la fiabilidad selectiva, la discriminación de errores y la calibración frente a utilizar solamente la confianza**, cuando el análisis se restringe al umbral operativo predefinido `score >= 0.20`. La fusión reduce el riesgo acumulado y permite mantener una cobertura mayor bajo restricciones de riesgo, lo que demuestra que la estabilidad y calidad de la localización aportan información útil para identificar detecciones potencialmente inseguras.

No obstante, la evidencia no respalda una mejora general en todas las dimensiones formuladas por la RQ. Frente a un control no espacial de igual capacidad, la fusión mejora significativamente AURC y Brier, pero no AUROC. Además, el reordenamiento espacial reduce el mAP COCO y el beneficio no es uniforme: varias categorías relevantes para seguridad, incluidas bicicleta, motocicleta y rider, presentan peores resultados de discriminación. El criterio confirmatorio global, que exigía seis de seis mejoras significativas, no se cumple.

La respuesta final es, por tanto, **condicional y parcial**: la calidad espacial complementa eficazmente la confianza para estimar y gestionar el riesgo de detecciones ya producidas, pero no constituye por sí sola una mejora universal del ranking detector ni una garantía general de seguridad. Su uso más defendible es como señal adicional para rechazo selectivo, priorización o escalamiento dentro de un sistema ADAS, conservando controles específicos por categoría y sin asumir que recupera objetos omitidos o que generaliza fuera del detector y dominio evaluados.

## Artefactos de respaldo

- [Métricas completas](outputs/metrics.json)
- [Contrastes primarios](outputs/Table_RQ3_primary_inference.csv)
- [Tabla principal](outputs/Table_RQ3_main.csv)
- [Ranking COCO](outputs/Table_RQ3_detector_ranking.csv)
- [Sensibilidad al umbral](outputs/Table_RQ3_threshold_sensitivity.csv)
- [Sensibilidad a pases Monte Carlo](outputs/Table_RQ3_mc_pass_sensitivity.csv)
- [Resultados por subgrupo](outputs/Table_RQ3_subgroups.csv)
- [Manifiesto del informe](outputs/report_manifest.json)

---

## English

### Research question

**How does fusing classification confidence with spatial localization quality improve reliability, ranking, and calibration for pinned GroundingDINO Swin-T detections on frozen BDD100K driving scenes?**

### Scope of the evidence

These results correspond to the complete confirmatory execution of the protocol frozen on July 31, 2026. The experiment used GroundingDINO Swin-T with the predefined checkpoint, prompts, seeds, partitions, and operating threshold. The interpretation is limited to detections conditioned on the model having produced a proposal; it therefore does not evaluate missed objects or demonstrate generalization to other detectors, datasets, or domains.

The experimental partition was:

| Partition | Images | Extracted detections | Purpose |
|---|---:|---:|---|
| Training | 5,600 | 1,195,746 | Estimator fitting |
| Validation | 2,400 | 510,987 | Group-independent selection and calibration |
| Confirmatory test | 1,992 | 422,843 | Final evaluation |

The primary analysis used **47,805 detections** with classification confidence `score >= 0.20`, as specified before observing the test results. Detection-error prevalence in this operating universe was 0.5685.

### Main results

#### Reliability, error discrimination, and calibration

The primary fusion, `product_fusion`, multiplicatively combines classification confidence with an estimated spatial-quality score. It was compared with confidence alone and with a capacity-matched non-spatial control.

| Method | AUROC ↑ | AUPRC ↑ | AURC ↓ | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Confidence | 0.7737 | 0.7834 | 0.3487 | 0.1880 | 0.5606 | 0.00955 |
| Non-spatial control | 0.8046 | 0.8308 | 0.3316 | 0.1767 | 0.5312 | 0.00927 |
| Spatial product fusion | **0.8056** | 0.8194 | **0.3261** | **0.1749** | **0.5277** | **0.00850** |

Against confidence alone, the fusion produced confirmatory improvements in all three primary metrics:

| Primary metric | Fusion improvement | 95% bootstrap CI | Holm-adjusted p | Decision |
|---|---:|---:|---:|---|
| AUROC | +0.03190 | [0.02873, 0.03504] | 0.002999 | Significant improvement |
| AURC | +0.02257 | [0.02069, 0.02439] | 0.002999 | Significant improvement |
| Brier | +0.01313 | [0.01192, 0.01428] | 0.002999 | Significant improvement |

For AURC and Brier, a positive improvement in the table denotes a reduction in a metric for which lower values are better. In relative terms, fusion reduced AURC by approximately 6.5% and Brier score by approximately 7.0% compared with confidence.

The evidence was more limited against the capacity-matched non-spatial control:

| Primary metric | Fusion improvement | 95% bootstrap CI | Holm-adjusted p | Decision |
|---|---:|---:|---:|---|
| AUROC | +0.00105 | [-0.00176, 0.00393] | 0.2409 | Not significant |
| AURC | +0.00553 | [0.00390, 0.00717] | 0.002999 | Significant improvement |
| Brier | +0.00183 | [0.00079, 0.00287] | 0.002999 | Significant improvement |

Overall, **five of the six primary comparisons** survived the Holm correction at `alpha = 0.05`. The predefined overall confirmatory criterion required all six comparisons to be favorable and significant. The correct result is therefore:

```text
nominal_calibration_criterion_met = true
nominal_ranking_criterion_met     = false
success_criterion_met             = false
```

This result does not imply an absence of benefit. It shows that fusion provides clear value over confidence alone and improves selective risk and Brier score relative to the capacity control, but it does not establish an additional AUROC advantage over that non-spatial control.

#### Selective risk

Fusion retained a larger proportion of detections while satisfying low-risk constraints:

| Risk constraint | Confidence | Non-spatial control | Spatial fusion |
|---|---:|---:|---:|
| Coverage at risk <= 0.05 | 0.42% | 0.52% | **2.60%** |
| Coverage at risk <= 0.10 | 2.43% | 7.42% | **10.84%** |
| Coverage at risk <= 0.20 | 22.62% | 26.12% | **28.03%** |

When retaining the 50% least uncertain detections, risk also decreased from 0.3682 with confidence to 0.3434 with fusion. This supports the use of spatial quality for selective rejection, review prioritization, or escalation to a more conservative subsystem.

#### Standard detector ranking

The benefit in error discrimination did not translate into better COCO detection ranking. Re-ranking all 422,843 proposals with the fusion reduced the standard detection metrics:

| Method | mAP@[.50:.95] ↑ | AP50 ↑ | AP75 ↑ |
|---|---:|---:|---:|
| Confidence | **0.2121** | **0.3693** | **0.2038** |
| Non-spatial control | 0.2094 | 0.3656 | 0.2013 |
| Spatial fusion | 0.2031 | 0.3508 | 0.1988 |

Fusion reduced mAP by 0.0090 absolute points, approximately 4.2% relative to confidence. This finding does not contradict the AUROC and AURC improvements: the primary metrics measure error-risk ordering within the `score >= 0.20` operating universe, whereas COCO AP measures class-specific ranking and the precision–recall curve across all proposals. Consequently, fusion should not be claimed to improve detector ranking in a general sense.

### Evidence about spatial quality

The taxonomy of the 47,805 operating detections was:

| Outcome | Detections | Proportion |
|---|---:|---:|
| Well localized and true positive | 20,627 | 43.15% |
| Well localized but detection error | 4,604 | 9.63% |
| Poorly localized | 22,574 | 47.22% |

There were 27,178 detection errors in total. Of these, 22,574, or approximately **83.1%**, were poorly localized detections. This confirms that spatial quality captures a relevant error mechanism that classification confidence alone does not adequately represent. Mean class-agnostic IoU was 0.4674.

The spatial-quality estimator achieved an AUROC of 0.7432 for identifying correct localization at IoU 0.50 and an AUROC of 0.7944 under the stricter IoU 0.75 criterion. The spatial signal is therefore informative, although it is not sufficient to replace semantic confidence.

### Robustness and sensitivity analyses

#### Number of Monte Carlo passes

Performance improved gradually as the number of Monte Carlo passes increased:

| MC passes | AUROC ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
|---:|---:|---:|---:|---:|
| 2 | 0.7990 | 0.3318 | 0.1784 | 0.00797 |
| 5 | 0.8031 | 0.3276 | 0.1761 | **0.00751** |
| 10 | **0.8056** | **0.3261** | **0.1749** | 0.00850 |

The AUROC, AURC, and Brier trends favor ten passes, while ECE shows small non-monotonic variation. Five passes provide results close to ten, but the confirmatory analysis must retain the predefined ten-pass setting.

#### Confidence threshold

Fusion utility depended on the operating regime:

| Threshold | Rows | Confidence AUROC | Fusion AUROC | Confidence AURC | Fusion AURC | Confidence Brier | Fusion Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 422,843 | **0.9144** | 0.9033 | 0.7943 | **0.7935** | **0.0401** | 0.0435 |
| 0.10 | 152,317 | 0.8586 | **0.8621** | 0.6142 | **0.6073** | 0.0963 | **0.0924** |
| 0.20 | 47,805 | 0.7737 | **0.8056** | 0.3487 | **0.3261** | 0.1880 | **0.1749** |
| 0.30 | 22,421 | 0.7093 | **0.7630** | 0.2090 | **0.1790** | 0.2004 | **0.1849** |

At the 0.05 threshold, fusion worsened AUROC and Brier, although it marginally improved AURC. The main conclusion is therefore restricted to the predefined 0.20 operating threshold and should not be extrapolated to all low-confidence proposals.

### Driving scenarios and subgroups

Descriptively, without separate confirmatory tests for each subgroup, fusion simultaneously improved AUROC, AURC, and Brier over confidence in 18 of 26 estimable subgroups. Particularly clear improvements were observed in residential scenes, parking lots, daytime driving, overcast weather, and small objects.

The effect was not uniform across categories:

| Category | Confidence AUROC | Fusion AUROC | Change |
|---|---:|---:|---:|
| Car | 0.7977 | **0.8225** | +0.0248 |
| Person | 0.8122 | **0.8164** | +0.0042 |
| Traffic sign | 0.7804 | **0.7959** | +0.0155 |
| Bicycle | **0.8624** | 0.8292 | -0.0332 |
| Bus | **0.9230** | 0.8897 | -0.0333 |
| Motorcycle | **0.8691** | 0.8198 | -0.0493 |
| Rider | **0.5420** | 0.5085 | -0.0335 |
| Truck | **0.8276** | 0.7996 | -0.0280 |

The deterioration for bicycle, motorcycle, and rider is especially relevant to road safety. These findings preclude recommending a universal policy that replaces confidence with fusion without category-specific safeguards and additional validation for vulnerable road users.

### Integrity and reproducibility

The post-run audit confirmed:

- 9,992 of 9,992 feature/image shard pairs with valid hashes.
- Reuse of all 9,992 shared shards and zero duplicated GPU inference during the RQ3 requests.
- Ten of ten model artifacts with valid SHA-256 hashes.
- Eighteen of eighteen report artifacts with valid SHA-256 hashes.
- Zero shared groups between selection and calibration.
- Zero overlap among the scientific training, validation, and test partitions.
- 2,000 bootstrap repetitions per analysis and 96,000 bootstrap records in total.
- Forty-nine automated tests passed.
- All required point metrics were finite.

No evidence of partial execution, corruption, partition contamination, or reuse of RQ1/RQ2 models was found.

### Conclusion and answer to the RQ

For GroundingDINO Swin-T on the frozen BDD100K partitions, fusing classification confidence with estimated spatial quality **significantly improves selective reliability, error discrimination, and calibration over confidence alone** when the analysis is restricted to the predefined `score >= 0.20` operating threshold. Fusion reduces accumulated risk and permits greater coverage under fixed-risk constraints, showing that localization stability and quality provide useful information for identifying potentially unsafe detections.

However, the evidence does not support a general improvement across every dimension of the RQ. Against a capacity-matched non-spatial control, fusion significantly improves AURC and Brier score but not AUROC. Spatial re-ranking also reduces COCO mAP, and the benefit is not uniform: several safety-relevant categories, including bicycle, motorcycle, and rider, show worse discrimination. The overall confirmatory criterion, which required six of six significant improvements, is therefore not met.

The final answer is consequently **conditional and partial**: spatial quality effectively complements confidence for estimating and managing the risk of detections that have already been produced, but it is not a universal improvement to detector ranking or a general safety guarantee. Its most defensible use is as an additional signal for selective rejection, prioritization, or escalation within an ADAS pipeline, with category-specific safeguards and without assuming that it recovers missed objects or generalizes beyond the evaluated detector and domain.

### Supporting artifacts

- [Complete metrics](outputs/metrics.json)
- [Primary comparisons](outputs/Table_RQ3_primary_inference.csv)
- [Main table](outputs/Table_RQ3_main.csv)
- [COCO ranking](outputs/Table_RQ3_detector_ranking.csv)
- [Threshold sensitivity](outputs/Table_RQ3_threshold_sensitivity.csv)
- [Monte Carlo pass sensitivity](outputs/Table_RQ3_mc_pass_sensitivity.csv)
- [Subgroup results](outputs/Table_RQ3_subgroups.csv)
- [Report manifest](outputs/report_manifest.json)
