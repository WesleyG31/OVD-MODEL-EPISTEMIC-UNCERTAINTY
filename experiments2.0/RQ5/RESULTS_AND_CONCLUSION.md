# RQ5 confirmatory results and conclusion / Resultados y conclusión confirmatorios de RQ5

Final evaluation date: 2026-08-02
Evidence status: `confirmatory_evaluation`
Frozen outcome: `fail`

## English

### Research question and claim boundary

RQ5 asks how fused epistemic uncertainty and calibrated detection outputs can
be integrated into an `accept`/`defer` decision layer for risk-aware selective
perception under ADAS latency constraints.

The evaluated policy isotonic-calibrates detector-confidence and epistemic-
uncertainty error estimates on group-disjoint validation data, combines them
with a fixed 0.5/0.5 late fusion, multiplies the fused error probability by a
prespecified criticality weight, and selects the acceptance threshold on a
separate validation fold at criticality-weighted risk <= 0.10. The
confirmatory claim is limited to detection-conditioned selective perception
and an offline latency-quality frontier. `Defer` denotes a request for an
external fallback; fallback outcomes and missed objects are not evaluated.

### Confirmatory population

- All 1,992 frozen confirmatory test images were processed.
- The detector produced 422,843 extracted detections and 47,805 operational
  candidates at the frozen score threshold 0.20.
- A total of 1,991 images contributed at least one operational candidate; one
  processed image had no detection above the threshold.
- Train, validation, and test image/sequence groups were disjoint.
- The six confirmatory comparisons used 2,000 paired image/sequence-cluster
  bootstrap repetitions each and one Holm family at alpha 0.05.

### Main method results

Lower weighted AURC, Brier, and operating risk are better; higher coverage is
better.

| Method | Weighted AURC | Coverage at weighted risk 0.10 | Operating coverage | Operating criticality-mass coverage | Operating weighted risk | Brier | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated confidence | 0.36684 | 0.01323 | 0.00498 | 0.00476 | 0.06434 | 0.18810 | 0.77347 |
| `flat_joint` | **0.34626** | 0.07128 | 0.07968 | 0.06687 | 0.09554 | 0.16940 | 0.82170 |
| Risk-aware fusion MC02 (primary) | 0.34818 | **0.08328** | 0.07018 | 0.05954 | 0.09744 | 0.17255 | 0.81433 |
| Risk-aware fusion MC05 (sensitivity) | 0.34519 | 0.08653 | 0.09007 | 0.07806 | 0.09075 | 0.16964 | 0.82062 |
| Risk-aware fusion MC10 (sensitivity) | 0.34324 | 0.10030 | 0.10549 | 0.09148 | 0.09507 | **0.16785** | **0.82411** |
| Uncertainty only | 0.34743 | 0.05674 | 0.06252 | 0.05354 | 0.09833 | 0.17085 | 0.81836 |
| Unweighted late fusion | 0.35385 | 0.08355 | 0.04897 | 0.04613 | 0.07467 | 0.17255 | 0.81433 |

At its validation-selected operating point, the primary policy accepted 3,355
of 47,805 candidates (7.02%), covered 5.95% of total criticality mass, and
obtained weighted risk 0.09744. It deferred 44,450 candidates and accepted 348
erroneous detections.

### Confirmatory inference

Improvements are oriented so that positive values favor risk-aware MC02.

| Comparison | Improvement | 95% percentile interval | Holm-adjusted p | Frozen outcome |
|---|---:|---:|---:|---|
| Weighted AURC vs calibrated confidence | +0.01866 | [0.01747, 0.01986] | 0.00300 | Pass |
| Coverage at weighted risk 0.10 vs calibrated confidence | +0.07005 | [0.03610, 0.08541] | 0.00300 | Pass |
| Weighted AURC vs `flat_joint` | **-0.00192** | [-0.00275, -0.00110] | 1.00000 | **Fail** |
| Coverage at weighted risk 0.10 vs `flat_joint` | +0.01201 | [-0.01821, 0.01961] | 0.00600 | Pass by the frozen null-centered test; percentile interval crosses zero |

Brier non-inferiority used an absolute margin of 0.01 and belonged to the same
Holm family. It passed against calibrated confidence (improvement +0.01555,
95% interval [0.01479, 0.01636], adjusted p=0.00300) and against `flat_joint`
(improvement -0.00315, 95% interval [-0.00372, -0.00257], adjusted p=0.00300).
The latter means MC02 had a slightly worse Brier score than `flat_joint`, but
the degradation remained within the frozen non-inferiority margin.

The overall Holm family failed because risk-aware MC02 did not improve
weighted AURC over the capacity-matched `flat_joint` baseline. Accordingly,
the prespecified confirmatory result is `success_status: fail`. The conflict
between the centered-test decision and the percentile interval for coverage
versus `flat_joint` is retained and must be reported transparently.

### Latency and systems result

| Configuration | Mean ms/image | p95 ms/image | Estimated FPS | Meets 100 ms |
|---|---:|---:|---:|---|
| Decision layer only | 1.08 | 1.55 | — | Yes; passes the 5 ms decision gate |
| Deterministic detector, MC00 | 403.61 | 424.42 | 2.48 | No |
| Primary MC02 | 1,203.72 | 1,224.96 | 0.83 | No |
| MC05 | 2,403.89 | 2,446.18 | 0.42 | No |
| MC10 | 4,404.17 | 4,485.24 | 0.23 | No |

The decision layer itself passed the frozen p95 <= 5 ms gate. However, none of
the complete detector configurations met the 33.3, 50, or 100 ms budgets on
the recorded NVIDIA GeForce RTX 4060 Laptop GPU. MC02 and MC05 are linear
prefix estimates from measured synchronized timing components; MC10 contains
the directly measured ten-pass stochastic block. The evaluated pipeline must
therefore be described as an offline latency-quality frontier, not as a
real-time ADAS deployment.

### Prespecified sensitivities

Five and ten MC passes improved the descriptive quality metrics, with MC10
reaching weighted AURC 0.34324 and coverage 0.10030, but at approximately
4.40 seconds per image. These results cannot replace the frozen MC02 primary
analysis or justify post-confirmatory model selection.

The criticality-geometry sensitivity also showed better descriptive weighted
AURC for no geometry (0.34526) and half geometry (0.34627) than for the frozen
primary geometry (0.34818). Strong geometry reduced operating weighted risk to
0.07883 but lowered operating coverage to 0.04836. These are secondary
trade-offs and cannot be used to retune the confirmatory policy.

### Conclusion

The confirmatory evidence supports the feasibility of a lightweight,
calibrated `accept`/`defer` layer and shows that fused uncertainty provides a
substantial selective-perception benefit over calibrated detector confidence
alone. Nevertheless, the proposed risk-aware MC02 construction did not
outperform a capacity-matched flat model in weighted AURC, so the data do not
support the claim that the proposed factorized fusion provides an advantage
beyond an equally expressive joint predictor. Moreover, although decision
overhead was below 5 ms, GroundingDINO plus MC inference was far outside all evaluated
real-time budgets. RQ5 is therefore answered with a mixed/negative result: the
decision mechanism is technically viable and informative, but the complete
method neither satisfied the frozen superiority rule nor achieved real-time
ADAS perception on the recorded hardware.

### Limitations and interpretation rules

- Evaluation is conditioned on produced detections and does not quantify
  false-negative objects, fallback quality, sensor fusion, or vehicle-level
  safety.
- The evidence covers one GroundingDINO Swin-T checkpoint, ten frozen BDD
  prompts, one BDD100K partition, and the recorded hardware.
- MC02/MC05 total latency is estimated from measured components rather than an
  online deadline benchmark.
- MC10, alternative criticality weights, thresholds, and subgroups are
  sensitivities only. They must not replace the frozen primary result.
- No confirmatory retuning is permitted. The negative result must be retained.

### Reproducibility and correction record

The post-confirmatory correction changed only the secondary
`weighted_risk_at_0.50` through `weighted_risk_at_0.90` operating points so
that they are indexed by cumulative criticality mass. It did not change any
primary metric, bootstrap comparison, Holm decision, Brier gate, latency
claim, or conclusion.

- Shared extraction fingerprint:
  `94bb3675c7f700e8123ee9db9fcc4d7502e25991f72d99f8fde8282dbd33a710`
- Evaluation source SHA-256:
  `31994059b0e0122e65210edc074154d487d41f80caf28b32f27c78b91b2736ff`
- Analysis fingerprint:
  `f0fe747f22f9191321cae19d0cd64b27f3bf84333b4e791429665c227598d211`
- Final `metrics.json` SHA-256:
  `2dd2be4ed973bce839964048736aa537a4df91428f50da07e1100e410ae33310`

The authoritative machine-readable evidence is in `outputs/metrics.json`,
with artifact hashes in `outputs/report_manifest.json`. The correction history
is documented in `POST_CONFIRMATORY_CORRECTIONS.md`.

---

## Español

### Pregunta de investigación y límite del claim

RQ5 pregunta cómo integrar incertidumbre epistémica fusionada y salidas
calibradas del detector en una capa de decisión `aceptar`/`diferir` para
percepción selectiva sensible al riesgo bajo restricciones de latencia ADAS.

La política evaluada calibra mediante isotónica las estimaciones de error de
confianza e incertidumbre epistémica usando datos de validación sin grupos
compartidos, las combina mediante una fusión tardía fija 0,5/0,5, multiplica la
probabilidad de error fusionada por un peso de criticidad preespecificado y
selecciona el umbral de aceptación en otro fold de validación con riesgo
ponderado por criticidad <= 0,10. El claim confirmatorio se limita a percepción
selectiva condicionada a detecciones y a una frontera offline de calidad-
latencia. `Diferir` solicita un fallback externo; no se evalúan sus resultados
ni los objetos omitidos por el detector.

### Población confirmatoria

- Se procesaron las 1.992 imágenes del test confirmatorio congelado.
- El detector produjo 422.843 detecciones extraídas y 47.805 candidatos
  operativos con el threshold congelado 0,20.
- Un total de 1.991 imágenes aportó al menos un candidato operativo; una imagen
  procesada no tuvo detecciones sobre el threshold.
- Los grupos de imagen/secuencia de train, validation y test no se solapan.
- Las seis comparaciones confirmatorias usaron 2.000 repeticiones bootstrap
  pareadas por imagen/secuencia y una familia Holm con alfa 0,05.

### Resultados principales por método

Menor AURC ponderado, Brier y riesgo operativo son mejores; mayor cobertura es
mejor.

| Método | AURC ponderado | Cobertura a riesgo ponderado 0,10 | Cobertura operativa | Cobertura operativa de masa crítica | Riesgo ponderado operativo | Brier | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Confianza calibrada | 0,36684 | 0,01323 | 0,00498 | 0,00476 | 0,06434 | 0,18810 | 0,77347 |
| `flat_joint` | **0,34626** | 0,07128 | 0,07968 | 0,06687 | 0,09554 | 0,16940 | 0,82170 |
| Fusión sensible al riesgo MC02 (primaria) | 0,34818 | **0,08328** | 0,07018 | 0,05954 | 0,09744 | 0,17255 | 0,81433 |
| Fusión sensible al riesgo MC05 (sensibilidad) | 0,34519 | 0,08653 | 0,09007 | 0,07806 | 0,09075 | 0,16964 | 0,82062 |
| Fusión sensible al riesgo MC10 (sensibilidad) | 0,34324 | 0,10030 | 0,10549 | 0,09148 | 0,09507 | **0,16785** | **0,82411** |
| Solo incertidumbre | 0,34743 | 0,05674 | 0,06252 | 0,05354 | 0,09833 | 0,17085 | 0,81836 |
| Fusión tardía sin criticidad | 0,35385 | 0,08355 | 0,04897 | 0,04613 | 0,07467 | 0,17255 | 0,81433 |

En el punto operativo elegido únicamente con validation, la política primaria
aceptó 3.355 de 47.805 candidatos (7,02%), cubrió 5,95% de la masa total de
criticidad y obtuvo riesgo ponderado 0,09744. Difirió 44.450 candidatos y
aceptó 348 detecciones erróneas.

### Inferencia confirmatoria

Las mejoras se orientan de modo que valores positivos favorecen MC02 sensible
al riesgo.

| Comparación | Mejora | Intervalo percentil 95% | p ajustado Holm | Resultado congelado |
|---|---:|---:|---:|---|
| AURC ponderado frente a confianza calibrada | +0,01866 | [0,01747, 0,01986] | 0,00300 | Pasa |
| Cobertura a riesgo ponderado 0,10 frente a confianza calibrada | +0,07005 | [0,03610, 0,08541] | 0,00300 | Pasa |
| AURC ponderado frente a `flat_joint` | **-0,00192** | [-0,00275, -0,00110] | 1,00000 | **Falla** |
| Cobertura a riesgo ponderado 0,10 frente a `flat_joint` | +0,01201 | [-0,01821, 0,01961] | 0,00600 | Pasa según el test centrado congelado; el intervalo percentil cruza cero |

La no inferioridad Brier usó margen absoluto 0,01 y perteneció a la misma
familia Holm. Pasó frente a confianza calibrada (mejora +0,01555, intervalo 95%
[0,01479, 0,01636], p ajustado=0,00300) y frente a `flat_joint` (mejora
-0,00315, intervalo 95% [-0,00372, -0,00257], p ajustado=0,00300). Este último
resultado significa que MC02 tuvo Brier ligeramente peor que `flat_joint`,
pero la degradación permaneció dentro del margen de no inferioridad congelado.

La familia Holm global falló porque MC02 sensible al riesgo no mejoró AURC
ponderado frente al baseline `flat_joint` de capacidad equivalente. Por tanto,
el resultado confirmatorio preespecificado es `success_status: fail`. Se
conserva y debe reportarse transparentemente la discordancia entre la decisión
del test centrado y el intervalo percentil para cobertura frente a
`flat_joint`.

### Latencia y resultado de sistema

| Configuración | Media ms/imagen | p95 ms/imagen | FPS estimados | Cumple 100 ms |
|---|---:|---:|---:|---|
| Solo capa de decisión | 1,08 | 1,55 | — | Sí; pasa el gate decisional de 5 ms |
| Detector determinista, MC00 | 403,61 | 424,42 | 2,48 | No |
| MC02 primario | 1.203,72 | 1.224,96 | 0,83 | No |
| MC05 | 2.403,89 | 2.446,18 | 0,42 | No |
| MC10 | 4.404,17 | 4.485,24 | 0,23 | No |

La capa decisional pasó el gate congelado p95 <= 5 ms. Sin embargo, ninguna
configuración completa del detector cumplió los presupuestos de 33,3, 50 o
100 ms en la GPU NVIDIA GeForce RTX 4060 Laptop registrada. MC02 y MC05 son
estimaciones lineales de prefijo basadas en componentes temporales medidos;
MC10 contiene el bloque estocástico de diez pasadas medido directamente. El
pipeline debe describirse como frontera offline calidad-latencia y no como
despliegue ADAS en tiempo real.

### Sensibilidades preespecificadas

Cinco y diez pasadas MC mejoraron descriptivamente las métricas de calidad;
MC10 alcanzó AURC ponderado 0,34324 y cobertura 0,10030, pero necesitó
aproximadamente 4,40 segundos por imagen. Estos resultados no pueden sustituir
el análisis primario MC02 congelado ni justificar selección posconfirmatoria.

La sensibilidad de geometría de criticidad también obtuvo mejor AURC
ponderado sin geometría (0,34526) y con media geometría (0,34627) que con la
geometría primaria congelada (0,34818). Geometría fuerte redujo el riesgo
ponderado operativo a 0,07883, pero redujo cobertura operativa a 0,04836. Son
trade-offs secundarios y no deben usarse para reajustar la política
confirmatoria.

### Conclusión

La evidencia confirmatoria respalda la factibilidad de una capa ligera y
calibrada de `aceptar`/`diferir`, y muestra que la incertidumbre fusionada
ofrece una mejora sustancial de percepción selectiva frente a usar únicamente
la confianza calibrada del detector. Sin embargo, la construcción MC02
sensible al riesgo no superó a un modelo plano de capacidad equivalente en
AURC ponderado; por ello, los datos no respaldan que la fusión factorizada
propuesta aporte una ventaja frente a un predictor conjunto igualmente
expresivo. Además, aunque el overhead decisional fue menor que 5 ms,
GroundingDINO con inferencia MC quedó muy fuera de todos los presupuestos de
tiempo real evaluados. RQ5 queda respondida con un resultado mixto/negativo: el
mecanismo decisional es técnicamente viable e informativo, pero el método
completo no cumplió la regla congelada de superioridad ni logró percepción
ADAS en tiempo real en el hardware registrado.

### Limitaciones y reglas de interpretación

- La evaluación está condicionada a detecciones producidas y no cuantifica
  objetos falsos negativos, calidad del fallback, fusión sensorial ni seguridad
  a nivel del vehículo.
- La evidencia cubre un checkpoint GroundingDINO Swin-T, diez prompts BDD
  congelados, una partición BDD100K y el hardware registrado.
- La latencia total MC02/MC05 se estima desde componentes medidos y no mediante
  un benchmark online con deadlines.
- MC10, pesos alternativos de criticidad, thresholds y subgrupos son sólo
  sensibilidades. No deben sustituir el resultado primario congelado.
- No se permite tuning confirmatorio. El resultado negativo debe conservarse.

### Reproducibilidad y registro de correcciones

La corrección posconfirmatoria modificó únicamente los puntos operativos
secundarios `weighted_risk_at_0.50` a `weighted_risk_at_0.90` para indexarlos
por masa acumulada de criticidad. No modificó métricas primarias, comparaciones
bootstrap, decisiones Holm, gate Brier, claim de latencia ni conclusión.

- Fingerprint de extracción compartida:
  `94bb3675c7f700e8123ee9db9fcc4d7502e25991f72d99f8fde8282dbd33a710`
- SHA-256 del evaluador:
  `31994059b0e0122e65210edc074154d487d41f80caf28b32f27c78b91b2736ff`
- Fingerprint del análisis:
  `f0fe747f22f9191321cae19d0cd64b27f3bf84333b4e791429665c227598d211`
- SHA-256 final de `metrics.json`:
  `2dd2be4ed973bce839964048736aa537a4df91428f50da07e1100e410ae33310`

La evidencia machine-readable autoritativa está en `outputs/metrics.json` y
los hashes de artefactos en `outputs/report_manifest.json`. El historial de
correcciones está documentado en `POST_CONFIRMATORY_CORRECTIONS.md`.
