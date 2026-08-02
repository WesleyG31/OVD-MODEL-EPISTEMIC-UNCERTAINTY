# RQ4 Results and Conclusion / Resultados y conclusión de RQ4

Confirmatory evaluation date / Fecha de evaluación confirmatoria: 2026-08-02
Frozen protocol date / Fecha del protocolo congelado: 2026-07-31

---

## English

### Research question

> To what extent does multi-level post-hoc calibration—combining class-level,
> localization-level, and uncertainty-level calibration—enhance detection
> reliability under prespecified BDD100K domain shifts?

### Confirmatory population and analysis

The confirmatory partition contained 1,992 images and 1,992 source sequences.
At the frozen score threshold of 0.20, there were 47,805 operational
detections, of which 42,716 detections from 1,821 sequences belonged to the
primary shifted-domain subset. Error prevalence in this subset was 56.95%.

All component fitting, hyperparameter selection, and final isotonic
calibration used only the frozen source domain (`daytime + clear + city
street`). The primary analysis used 2,000 paired sequence-cluster bootstrap
replicates and Holm correction over the six prespecified comparisons. A
positive contrast is defined as `baseline metric - multilevel metric`, so a
positive value favors multi-level calibration for Brier, NLL, and AURC.

### Primary results

| Method | Brier ↓ | NLL ↓ | AURC ↓ | ECE ↓ | AUROC ↑ | AUPRC ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated confidence | 0.190139 | 0.568303 | 0.351700 | 0.027128 | 0.771093 | 0.781912 |
| Flat joint control | **0.131982** | **0.414290** | **0.273671** | **0.012086** | **0.891181** | **0.910600** |
| Multi-level calibration | 0.144282 | 0.446514 | 0.285472 | 0.016730 | 0.870754 | 0.891081 |

Multi-level calibration substantially outperformed calibrated confidence:

| Metric | Absolute improvement | Relative improvement | 95% paired bootstrap interval | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Brier | +0.045857 | +24.12% | [0.043793, 0.047886] | 0.002999 |
| NLL | +0.121789 | +21.43% | [0.116085, 0.127251] | 0.002999 |
| AURC | +0.066228 | +18.83% | [0.063256, 0.069095] | 0.002999 |

However, multi-level calibration was consistently worse than the flat joint
control, despite both methods receiving the same 21 input features and having
comparable capacity (32 versus 30 fitted coefficients):

| Metric | Contrast vs. flat control | Relative improvement | 95% paired bootstrap interval | One-sided Holm p for multi-level superiority |
|---|---:|---:|---:|---:|
| Brier | -0.012301 | -9.32% | [-0.013504, -0.011141] | 1.000 |
| NLL | -0.032224 | -7.78% | [-0.035912, -0.028567] | 1.000 |
| AURC | -0.011801 | -4.31% | [-0.012974, -0.010590] | 1.000 |

Because the frozen success rule required multi-level calibration to improve
all three metrics over both baselines, the confirmatory success criterion was
**not met**. This is a mixed result: all three comparisons against calibrated
confidence were favorable and survived multiplicity correction, whereas all
three comparisons against the same-feature flat control were unfavorable.

### Secondary and robustness results

- Increasing the MC budget produced small but consistent improvements:

  | MC passes | Brier ↓ | NLL ↓ | AURC ↓ |
  |---:|---:|---:|---:|
  | 2 | 0.146257 | 0.453706 | 0.289146 |
  | 5 | 0.144941 | 0.449651 | 0.286629 |
  | 10 | 0.144282 | 0.446514 | 0.285472 |

- Multi-level calibration remained better than calibrated confidence at every
  prespecified score threshold (0.05, 0.10, 0.20, and 0.30). The flat model was
  generally strongest; at threshold 0.05, multi-level had lower NLL but
  slightly worse Brier and AURC than the flat model.
- Reliability deteriorated under shift. For multi-level calibration, the
  shifted-minus-reference gaps were +0.011609 Brier, +0.032289 NLL, and
  +0.020760 AURC. The time-of-day shift was the most difficult major axis
  (Brier 0.153658, NLL 0.471006, AURC 0.310275).
- The flat joint model was better than the multi-level model in the reference
  domain, all three nonzero shift-severity strata, all object-size strata, and
  most sufficiently populated attribute strata.
- The `class_localization` ablation (Brier 0.141870, NLL 0.437148, AURC
  0.281140) descriptively outperformed the complete multi-level product. This
  suggests that multiplying by the uncertainty-level component did not add
  reliability in this experiment. This ablation result is secondary and was
  not part of the confirmatory multiplicity-controlled family.

### Conclusion

Within the frozen GroundingDINO and BDD100K setting, multi-level post-hoc
calibration clearly enhances reliability relative to confidence-based
calibration under internal covariate shifts. It reduces Brier score, NLL, and
AURC by approximately 24%, 21%, and 19%, respectively, relative to calibrated
confidence.

Nevertheless, the proposed factorized multi-level formulation does not provide
an advantage over a single flat logistic calibrator trained on the same
features. The flat control achieved better probability calibration and better
error ranking, with adverse multi-level contrasts whose 95% paired bootstrap
intervals were entirely below zero. The evidence therefore indicates that the
main benefit comes from jointly learning from class, localization, and
epistemic features, rather than from multiplying separately calibrated
reliability components.

Accordingly, the confirmatory hypothesis is not supported in its full form.
The scientifically appropriate conclusion is not that multi-level calibration
is ineffective, but that it improves a conventional confidence baseline while
failing to outperform the stronger same-information flat alternative. No
confirmatory retuning or test-set-driven modification should be performed.

### Claim boundary

These findings apply to the pinned detector, vocabulary, BDD100K partition,
and prespecified within-BDD attribute shifts. They do not establish
generalization to another dataset, sensor, geography, or open-set vocabulary.
The analysis concerns reliability of fixed detector candidates above the
operational score threshold; it does not recover false negatives or establish
ADAS safety.

---

## Español

### Pregunta de investigación

> ¿En qué medida la calibración post-hoc multinivel —combinando calibración a
> nivel de clase, localización e incertidumbre— mejora la fiabilidad de las
> detecciones bajo cambios de dominio BDD100K preespecificados?

### Población y análisis confirmatorio

La partición confirmatoria contenía 1.992 imágenes y 1.992 secuencias fuente.
Con el umbral congelado de score 0,20 se obtuvieron 47.805 detecciones
operacionales, de las cuales 42.716 detecciones pertenecientes a 1.821
secuencias formaron el subconjunto primario desplazado. La prevalencia de error
en este subconjunto fue 56,95%.

El ajuste de componentes, la selección de hiperparámetros y la calibración
isotónica final utilizaron exclusivamente el dominio fuente congelado
(`daytime + clear + city street`). El análisis primario empleó 2.000 réplicas
bootstrap pareadas por secuencia y corrección de Holm sobre las seis
comparaciones preespecificadas. Un contraste positivo se define como
`métrica del baseline - métrica multinivel`, por lo que valores positivos
favorecen al método multinivel en Brier, NLL y AURC.

### Resultados primarios

| Método | Brier ↓ | NLL ↓ | AURC ↓ | ECE ↓ | AUROC ↑ | AUPRC ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Confianza calibrada | 0,190139 | 0,568303 | 0,351700 | 0,027128 | 0,771093 | 0,781912 |
| Control plano conjunto | **0,131982** | **0,414290** | **0,273671** | **0,012086** | **0,891181** | **0,910600** |
| Calibración multinivel | 0,144282 | 0,446514 | 0,285472 | 0,016730 | 0,870754 | 0,891081 |

La calibración multinivel superó ampliamente a la confianza calibrada:

| Métrica | Mejora absoluta | Mejora relativa | Intervalo bootstrap pareado del 95% | p ajustado por Holm |
|---|---:|---:|---:|---:|
| Brier | +0,045857 | +24,12% | [0,043793; 0,047886] | 0,002999 |
| NLL | +0,121789 | +21,43% | [0,116085; 0,127251] | 0,002999 |
| AURC | +0,066228 | +18,83% | [0,063256; 0,069095] | 0,002999 |

Sin embargo, el método multinivel fue consistentemente peor que el control
plano, aunque ambos recibieron exactamente los mismos 21 features y tuvieron
capacidad comparable (32 frente a 30 coeficientes ajustados):

| Métrica | Contraste frente al control plano | Mejora relativa | Intervalo bootstrap pareado del 95% | p Holm unilateral para superioridad multinivel |
|---|---:|---:|---:|---:|
| Brier | -0,012301 | -9,32% | [-0,013504; -0,011141] | 1,000 |
| NLL | -0,032224 | -7,78% | [-0,035912; -0,028567] | 1,000 |
| AURC | -0,011801 | -4,31% | [-0,012974; -0,010590] | 1,000 |

Como la regla congelada exigía mejorar las tres métricas frente a ambos
baselines, el criterio de éxito confirmatorio **no se cumplió**. El resultado
es mixto: las tres comparaciones frente a confianza calibrada fueron
favorables y sobrevivieron la corrección por multiplicidad, mientras que las
tres comparaciones frente al control plano de mismos features fueron
desfavorables.

### Resultados secundarios y robustez

- Aumentar el presupuesto MC produjo mejoras pequeñas pero consistentes:

  | Pasadas MC | Brier ↓ | NLL ↓ | AURC ↓ |
  |---:|---:|---:|---:|
  | 2 | 0,146257 | 0,453706 | 0,289146 |
  | 5 | 0,144941 | 0,449651 | 0,286629 |
  | 10 | 0,144282 | 0,446514 | 0,285472 |

- La calibración multinivel fue mejor que la confianza calibrada en todos los
  umbrales de score preespecificados (0,05; 0,10; 0,20 y 0,30). El modelo plano
  fue generalmente el más fuerte; con umbral 0,05, el multinivel obtuvo menor
  NLL, pero Brier y AURC ligeramente peores que el plano.
- La fiabilidad empeoró bajo shift. Para el multinivel, los gaps
  desplazado-menos-referencia fueron +0,011609 en Brier, +0,032289 en NLL y
  +0,020760 en AURC. El cambio de hora del día fue el eje principal más difícil
  (Brier 0,153658; NLL 0,471006; AURC 0,310275).
- El modelo plano fue mejor que el multinivel en el dominio de referencia, en
  los tres niveles no nulos de severidad, en todos los tamaños de objeto y en
  la mayoría de los estratos de atributos con tamaño suficiente.
- La ablación `class_localization` (Brier 0,141870; NLL 0,437148; AURC
  0,281140) superó descriptivamente al producto multinivel completo. Esto
  sugiere que multiplicar por el componente de incertidumbre no añadió
  fiabilidad en este experimento. Esta ablación es secundaria y no formó parte
  de la familia confirmatoria controlada por multiplicidad.

### Conclusión

Dentro del escenario congelado de GroundingDINO y BDD100K, la calibración
post-hoc multinivel mejora claramente la fiabilidad respecto a la calibración
basada en confianza bajo cambios covariables internos. Frente a la confianza
calibrada, reduce aproximadamente 24% el Brier, 21% el NLL y 19% el AURC.

No obstante, la formulación multinivel factorizada propuesta no ofrece una
ventaja frente a un único calibrador logístico plano entrenado con los mismos
features. El control plano obtuvo mejor calibración probabilística y mejor
ordenamiento de errores; además, los intervalos bootstrap pareados del 95% de
los contrastes multinivel fueron completamente negativos. La evidencia indica
que el beneficio principal procede de aprender conjuntamente a partir de
features de clase, localización e incertidumbre epistémica, no de multiplicar
componentes de fiabilidad calibrados por separado.

En consecuencia, la hipótesis confirmatoria no queda respaldada en su forma
completa. La conclusión científicamente adecuada no es que la calibración
multinivel sea ineficaz, sino que mejora un baseline convencional de confianza
pero no supera a la alternativa plana más fuerte con la misma información. No
debe realizarse retuning confirmatorio ni modificar el método a partir del
test.

### Límite de la afirmación

Estos resultados se limitan al detector, vocabulario, partición BDD100K y
shifts internos preespecificados. No demuestran generalización a otro dataset,
sensor, geografía o vocabulario open-set. El análisis evalúa la fiabilidad de
candidatos fijos por encima del umbral operacional; no recupera falsos
negativos ni establece seguridad ADAS.

---

## Reproducibility and reporting note / Nota de reproducibilidad y reporte

The numerical source of truth for this document is
`RQ4/outputs/metrics.json`, whose inputs and outputs passed the recorded hash
checks. The current generated `Table_RQ4_primary_inference.csv` overwrites the
baseline method name with the numeric baseline value, and the generated figure
caption still labels the output as diagnostic. Those are reporting defects and
do not change the calculations or conclusions above.

La fuente numérica de este documento es `RQ4/outputs/metrics.json`, cuyas
entradas y salidas superaron las comprobaciones de hashes registradas. El
archivo generado `Table_RQ4_primary_inference.csv` sobrescribe actualmente el
nombre del baseline con su valor numérico y la leyenda generada todavía llama
diagnóstico al output. Son defectos de reporte que no cambian los cálculos ni
las conclusiones anteriores.
