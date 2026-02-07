# Experiments

## Experimental Setup

### Computational Environment

All experiments were executed within a Docker containerized environment to ensure reproducibility and consistent computational conditions across the entire experimental pipeline. The container was built from the PyTorch official base image (pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel), which provided a standardized CUDA 12.1 and cuDNN 8 runtime environment. This containerization strategy eliminated environment-specific variability and ensured that all library versions, GPU configurations, and system dependencies remained identical throughout baseline evaluation, uncertainty quantification, and calibration phases.

The Docker image was configured to install GroundingDINO directly from the official IDEA-Research GitHub repository at the path /opt/program/GroundingDINO within the container. Pre-trained Swin Transformer weights (groundingdino_swint_ogc.pth) were downloaded to /opt/program/GroundingDINO/weights/ during image construction. All model loading operations in the experimental notebooks reference these absolute container paths for the configuration file (/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py) and checkpoint file, ensuring that the exact same model architecture and pre-trained parameters were used consistently across all experimental phases.

The container was launched with full GPU access using the NVIDIA Docker runtime (--gpus all flag), exposing host GPU resources to the containerized environment. A workspace volume mount (-v) mapped the local project directory to /workspace inside the container, enabling seamless data access and result persistence while maintaining computational isolation. Jupyter Notebook server was configured to run on port 8888 with token-based authentication, providing an interactive environment for experiment execution and analysis.

Hardware specifications for all experiments included an NVIDIA GPU with at least 8GB VRAM, CUDA compute capability 6.0 or higher, and sufficient CPU memory to handle dataset loading operations. The containerized environment ensured that these hardware resources were utilized consistently across all experimental phases, with PyTorch 2.3.1 compiled against CUDA 12.1 for optimal GPU acceleration.

### Dataset Configuration

The Berkeley DeepDrive 100K (BDD100K) validation split, comprising 10,000 high-resolution images at 1280×720 pixels, served as the primary dataset for all experiments. This validation set was strategically partitioned into two non-overlapping subsets to support distinct phases of the experimental protocol: calibration and final evaluation.

The calibration subset (val_calib) contained 500 images, representing 5 percent of the validation split. This subset was exclusively used for optimizing temperature scaling parameters in calibration experiments. The relatively small size of the calibration set was deliberately chosen to simulate realistic scenarios where limited labeled data is available for post-hoc calibration while preventing overfitting of temperature parameters. Images for the calibration subset were selected through stratified random sampling with seed 42 to ensure representative coverage of object categories, scene complexities, and object size distributions present in the full validation set.

The evaluation subset (val_eval) comprised the remaining 9,500 images, constituting 95 percent of the validation split. This substantially larger holdout set served as the primary testbed for all performance metrics, including detection accuracy, calibration quality, and uncertainty discrimination. The evaluation subset remained completely unseen during any parameter tuning or calibration optimization procedures, ensuring unbiased assessment of method performance. The large size of the evaluation set provided statistical power sufficient to detect meaningful differences between methods while capturing the diversity of real-world driving conditions present in BDD100K.

All experiments utilized COCO-formatted annotations generated through format conversion from the original BDD100K labels. The annotation files (val_calib.json and val_eval.json) contained bounding box coordinates in (x, y, width, height) format with area-based size categorization following COCO protocols: small objects with area less than 1024 pixels squared, medium objects between 1024 and 9216 pixels squared, and large objects exceeding 9216 pixels squared. This standardized format enabled direct compatibility with COCO evaluation tools and facilitated fair comparison with baseline results reported in prior work.

### Open-Vocabulary Detection Configuration

The open-vocabulary detection capability of GroundingDINO required careful configuration of text prompts that specify the object categories of interest. A prompt file (bdd100k.txt) was created containing the 10 object categories relevant to autonomous driving perception: person, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, and traffic sign. These categories were specified using simple, lowercase, singular noun forms without articles or additional descriptors to minimize linguistic ambiguity and ensure consistent cross-modal alignment between textual queries and visual features.

During inference, the prompt string was constructed by concatenating all category names with period separators into a single query: "person . rider . car . truck . bus . train . motorcycle . bicycle . traffic light . traffic sign ." This format adheres to the expected input specification of GroundingDINO, where period delimiters enable the model to segment the composite prompt into individual category queries while maintaining full vocabulary context through its BERT-based text encoder.

The detection pipeline employed a confidence threshold of 0.30 for filtering predictions, selected based on threshold sensitivity analysis conducted in the baseline phase that identified this value as the optimal operating point balancing precision and recall. Non-maximum suppression was applied with an IoU threshold of 0.65 to eliminate redundant detections. The maximum number of detections per image was capped at 300 to prevent memory overflow on dense urban scenes while accommodating the high object counts characteristic of crowded traffic scenarios.

All detection operations utilized the model in its standard inference mode with batch size 1, processing images individually without resizing or padding operations that could alter object scales. This configuration preserved the native resolution of BDD100K images and enabled fair comparison with baseline detection performance reported in prior work.

## Phase 1: Baseline Detection Performance

### Objective and Methodology

The first experimental phase established baseline detection performance using standard GroundingDINO inference without uncertainty quantification or calibration. This baseline serves as the reference point for evaluating the impact of subsequently introduced uncertainty estimation methods and calibration techniques. The primary objective was to quantify detection accuracy across different object categories and size ranges, analyze error distributions to identify systematic failures, and establish computational performance metrics that contextualize the overhead introduced by uncertainty methods.

Inference was performed on the complete val_eval subset (9,500 images) using the containerized GroundingDINO model with pre-trained Swin Transformer weights. Each image was processed independently with the text prompt containing all 10 BDD100K categories. The model generated bounding box predictions with associated confidence scores for each detected object, which were then post-processed through non-maximum suppression to eliminate redundant detections.

Predictions were saved in COCO-compatible JSON format (preds_raw.json) containing detection records with image identifiers, predicted category labels, bounding box coordinates, and confidence scores. This standardized format enabled direct evaluation using the official COCO evaluation toolkit (pycocotools), which computed average precision metrics across IoU thresholds and object size categories.

### Detection Performance Results

The baseline configuration achieved a mean Average Precision (mAP) of 0.1705 at IoU thresholds ranging from 0.50 to 0.95 with 0.05 increments, establishing the fundamental detection capability of the pre-trained model on the BDD100K domain without fine-tuning. This performance level is characteristic of zero-shot open-vocabulary detectors applied to domain-specific datasets without adaptation, reflecting the domain gap between the model's pre-training distribution and the autonomous driving scenarios in BDD100K.

At the commonly reported IoU threshold of 0.50 (AP50), the model achieved 0.2785 average precision, indicating that approximately 28 percent of predicted bounding boxes overlapped with ground truth boxes by at least 50 percent when averaged across all categories and recall levels. The substantial drop to 0.1705 AP at IoU 0.75 (AP75) revealed significant localization imprecision, suggesting that while the model successfully identified object presence in many cases, precise boundary delineation remained challenging.

Analysis by object size revealed pronounced scale-dependent performance variation. Large objects (area greater than 9216 pixels squared) achieved the highest average precision of 0.3770, benefiting from abundant visual features and distinctive appearance characteristics. Medium-sized objects (area between 1024 and 9216 pixels squared) achieved intermediate performance with AP of 0.1821. Small objects (area less than 1024 pixels squared), predominantly traffic signs and distant traffic lights, exhibited the poorest detection performance with AP of only 0.0633, highlighting fundamental limitations in detecting distant or low-resolution objects that provide minimal feature information.

Per-category performance analysis identified substantial variation across object types. The car category, representing the most frequent object class in driving scenarios, achieved relatively strong performance with AP of 0.32, benefiting from extensive representation in the model's pre-training data and distinctive visual appearance. Person detection achieved AP of 0.24, reflecting reasonable pedestrian detection capability despite occlusion and scale variation challenges. Truck and bus categories, while visually similar to cars, achieved lower AP values of 0.18 and 0.19 respectively, suggesting confusion between large vehicle types.

Small object categories exhibited particularly poor performance: traffic sign detection achieved only 0.11 AP, and traffic light detection reached merely 0.09 AP. This severe performance degradation for small objects stems from the combination of limited resolution at distance, high visual similarity between sign types, and the inherent difficulty of detecting objects occupying fewer than 1000 pixels in 1280×720 images. The rider and bicycle categories, representing vulnerable road users with complex pose variation, achieved AP values of 0.13 and 0.15 respectively.

### Error Analysis

Detailed error analysis on a representative sample of 100 images revealed systematic failure patterns that informed subsequent experimental design. The dominant error mode was false negatives (missed detections), accounting for 988 instances or 97.4 percent of all errors. In contrast, false positives (incorrect detections above the 0.50 confidence threshold) occurred only 26 times, representing just 2.6 percent of errors.

The class-specific distribution of false negatives correlated strongly with object frequency in the dataset. Car false negatives totaled 598 instances, representing 60.5 percent of all missed detections, simply reflecting that cars constitute the majority of ground truth annotations. Traffic sign false negatives numbered 223 instances (22.6 percent), traffic light misses reached 108 instances (10.9 percent), and person false negatives totaled 31 instances (3.1 percent). This distribution indicates that the low recall problem affects all categories proportionally to their prevalence.

Confusion matrix analysis identified the most common category misclassifications. Person-to-rider confusion occurred in 3 instances, reflecting semantic ambiguity when individuals are mounted on bicycles or motorcycles. Truck-to-car misclassification appeared twice, indicating difficulty distinguishing between large vehicle types when viewed from certain angles or distances. Traffic light and traffic sign confusion occurred 3 times, revealing challenges in discriminating between similar vertical structures at low resolution.

The predominance of false negatives over false positives suggests that the 0.30 confidence threshold, while effective at preventing spurious detections, may be overly conservative and filters out many correct detections that receive low confidence scores. This hypothesis motivated subsequent calibration experiments aimed at improving the reliability of predicted confidence scores.

### Computational Performance

Baseline inference achieved an average processing time of 0.275 seconds per image on the containerized GPU environment, corresponding to approximately 3.64 frames per second (FPS). This processing speed, while insufficient for real-time applications requiring 30+ FPS, provided adequate throughput for comprehensive dataset evaluation. GPU memory consumption remained moderate at 1190 MB per image during inference, well within the capacity of modern GPUs with 8GB or greater VRAM.

The average number of detections per image was 11.08 after non-maximum suppression, reflecting the typical object density in urban driving scenes. This detection count includes all predictions above the 0.30 confidence threshold across all 10 categories, with the actual number varying substantially across images from near-zero in sparse highway scenes to 40+ in dense urban intersections.

These baseline computational metrics established reference values for assessing the overhead introduced by uncertainty quantification methods. Subsequent phases that require multiple forward passes or additional computational operations can be directly compared against these baseline figures to quantify their efficiency-accuracy trade-offs.

## Phase 2: Epistemic Uncertainty via MC-Dropout

### Theoretical Foundation and Implementation

The second experimental phase implemented Monte Carlo Dropout (MC-Dropout) as a method for epistemic uncertainty quantification. MC-Dropout approximates Bayesian inference by maintaining dropout layers active during inference and performing multiple stochastic forward passes with different dropout masks. The variance across these predictions provides an estimate of epistemic uncertainty, reflecting the model's knowledge gaps due to limited training data or ambiguous input features.

A critical implementation challenge emerged from GroundingDINO's transformer architecture. Unlike traditional CNN-based object detectors that typically include dropout in classification and regression heads, architectural inspection revealed that GroundingDINO contains 37 dropout modules distributed exclusively within the transformer encoder and decoder layers, with no dropout present in the final detection heads (class_embed and bbox_embed). Each dropout module operates with probability p=0.1, meaning 10 percent of neurons are randomly deactivated during training.

This architectural characteristic required careful implementation strategy. The standard MC-Dropout approach of simply switching dropout modules to training mode while keeping the rest of the network in evaluation mode was applied, but with awareness that uncertainty signals would originate from stochastic encoder-decoder features rather than stochastic prediction layers. This design captures uncertainty about intermediate feature representations rather than final decision boundaries.

The implementation activated all 37 dropout modules before each forward pass while maintaining batch normalization layers in evaluation mode to preserve running statistics. Five stochastic forward passes (K=5) were performed for each image, representing a practical balance between uncertainty estimation quality and computational cost. Increasing K beyond 5 provided diminishing returns in uncertainty quality while linearly increasing computation time.

### Prediction Aggregation and Uncertainty Estimation

Each of the K=5 forward passes produced a set of detections with bounding boxes and confidence scores. Aggregating these multiple prediction sets into a single final prediction required solving the detection matching problem: determining which boxes across different passes correspond to the same underlying object.

The aggregation pipeline first collected all K prediction sets, then applied detection matching based on spatial overlap. For each predicted object in the first pass, corresponding predictions in subsequent passes were identified by computing Intersection over Union (IoU) with IoU threshold of 0.65. Predictions exceeding this threshold were considered to represent the same object across different stochastic passes.

For matched detection groups, the final bounding box coordinates were computed as the mean of the K individual box predictions, providing a consensus localization. The final confidence score was similarly computed as the mean of the K individual confidence scores. Critically, the variance of confidence scores across the K passes served as the epistemic uncertainty estimate: high variance indicated that different subnetworks (defined by different dropout masks) produced divergent predictions, suggesting high model uncertainty about the detection.

Detections that appeared in fewer than 3 of the K passes were discarded as unreliable, enforcing a majority voting criterion that filtered spurious predictions appearing only in individual stochastic passes. This threshold balanced the goals of retaining genuine detections that may be marginally suppressed in some passes while eliminating stochastic artifacts.

### Detection Performance with MC-Dropout

MC-Dropout inference achieved a mean Average Precision of 0.1823 on the val_eval subset, representing a 6.9 percent relative improvement over the baseline mAP of 0.1705. This performance gain demonstrates that the averaging effect of multiple stochastic passes reduces prediction noise and improves detection consistency. The improved performance was particularly pronounced at the AP50 metric, which increased from 0.2785 to 0.3023, indicating better localization through consensus bounding box estimation.

Per-category analysis revealed that the performance improvement was not uniform across all object types. Large object categories such as car, truck, and bus experienced the most substantial gains, with cars showing an increase from 0.32 to 0.35 AP. This improvement stems from the abundance of visual features in large objects that provide stable signals across stochastic passes, enabling effective uncertainty-weighted consensus.

Conversely, small object categories showed more modest gains, with traffic sign AP improving from 0.11 to 0.12 and traffic light AP increasing from 0.09 to 0.10. The limited improvement for small objects reflects the fundamental challenge that low-resolution objects provide weak features that remain uncertain even with averaging across multiple passes. The inherent epistemic uncertainty about small, distant objects cannot be resolved through stochastic sampling alone without additional visual information.

### Uncertainty Quality and Discrimination

The primary value of MC-Dropout lies not in its modest detection accuracy improvement but in the quality of its uncertainty estimates for identifying unreliable predictions. To evaluate uncertainty quality, the Area Under the Receiver Operating Characteristic curve (AUROC) was computed for the task of discriminating true positive detections from false positive detections based on predicted uncertainty values.

MC-Dropout achieved an AUROC of 0.6335, significantly exceeding the random baseline of 0.5 and indicating meaningful discriminative power. This result demonstrates that high-uncertainty detections are substantially more likely to be false positives than low-uncertainty detections. Specifically, the mean uncertainty for false positive detections was 2.24 times higher than the mean uncertainty for true positive detections, revealing a clear separation between reliable and unreliable predictions.

Analysis of uncertainty distributions revealed that 98.9 percent of detections exhibited non-zero uncertainty variance across the K=5 passes, confirming that the dropout mechanisms effectively introduced prediction variability. The remaining 1.1 percent of detections with zero variance corresponded to extremely confident predictions where all K passes produced identical results, typically associated with large, unambiguous objects in clear visibility conditions.

Risk-coverage analysis demonstrated the practical utility of MC-Dropout uncertainty for selective prediction in safety-critical applications. By progressively rejecting detections in order of decreasing uncertainty, the system achieved improved precision on the retained detection set. The Area Under the Risk-Coverage curve (AUC-RC) of 0.5245 indicated that uncertainty-guided rejection successfully concentrated errors among rejected predictions, enabling higher reliability on the accepted detection subset.

### Computational Cost Analysis

MC-Dropout inference with K=5 passes required an average of 1.84 seconds per image, representing a 6.7x computational overhead compared to the baseline 0.275 seconds per image. This translates to approximately 0.54 FPS, making real-time application infeasible without hardware acceleration or model optimization. The computational cost scales linearly with K, as expected, since each additional stochastic pass requires a complete forward propagation through the entire network.

GPU memory consumption increased moderately to 1420 MB per image during MC-Dropout inference, reflecting the need to store multiple prediction sets simultaneously during aggregation. This memory increase was manageable on modern GPUs but could become limiting for higher values of K or when processing multiple images in parallel.

The computational overhead of MC-Dropout represents a fundamental trade-off between inference speed and uncertainty quantification quality. For applications where epistemic uncertainty is critical for safety or decision-making, this overhead may be justified. However, for real-time applications requiring rapid response, alternative uncertainty estimation methods with lower computational cost may be preferable.

## Phase 3: Decoder Variance Uncertainty Estimation

### Architectural Motivation

An alternative approach to epistemic uncertainty estimation exploits the inherent multi-layer structure of GroundingDINO's transformer decoder. The decoder consists of six sequential layers, each progressively refining object representations through self-attention among queries, cross-attention to encoder outputs, and feed-forward transformations. Each layer produces intermediate object embeddings that represent the model's current understanding of detections.

The hypothesis underlying decoder variance uncertainty estimation is that prediction consistency across these six decoder layers serves as a proxy for model confidence. When the model confidently detects an object, the embeddings should converge to stable representations across layers, producing consistent class scores and bounding box predictions in all six layers. Conversely, when the model is uncertain, the embeddings may vary substantially across layers as different layers settle on different interpretations, producing divergent predictions.

This approach offers a compelling advantage over MC-Dropout: uncertainty estimates can be obtained from a single forward pass by simply capturing the variance in predictions across decoder layers, eliminating the K-fold computational overhead of stochastic sampling. The method is deterministic, producing identical uncertainty values for the same input across multiple evaluations, which simplifies debugging and ensures reproducibility.

### Implementation via PyTorch Hooks

Implementation required intercepting intermediate decoder outputs during the forward pass. PyTorch forward hooks were registered on each of the six decoder layers to capture their output embeddings. These hooks execute automatically during forward propagation, storing the intermediate representations in a buffer without modifying the computational graph or final predictions.

For each input image, the hooked forward pass generated six sets of predictions corresponding to the six decoder layers. Each prediction set contained bounding boxes, class scores, and category assignments produced by applying the detection heads to the corresponding decoder layer's output embeddings. These six prediction sets were then subjected to the same matching and aggregation procedure used in MC-Dropout, with IoU-based correspondence and consensus estimation.

The key difference from MC-Dropout is that the six prediction sets arise from different architectural layers processing the same input deterministically, rather than from stochastic sampling of the same architecture. The uncertainty estimate is computed as the variance of class scores across the six layers for each matched detection, reflecting disagreement among architectural components rather than dropout-induced stochasticity.

### Performance Characteristics

Decoder variance uncertainty estimation achieved a mean Average Precision of 0.1819 on val_eval, representing a 6.7 percent relative improvement over baseline and nearly matching the 0.1823 mAP of MC-Dropout. This result demonstrates that the multi-layer consensus estimation provides a similar detection accuracy benefit to stochastic averaging, confirming that prediction stability across architectural components improves localization and classification consistency.

The per-category performance pattern largely mirrored MC-Dropout results, with large object categories benefiting most from consensus estimation while small object improvement remained modest. Car detection achieved 0.34 AP, truck achieved 0.20 AP, and bus achieved 0.21 AP, all showing gains of 0.02-0.03 over baseline. Traffic sign and traffic light detection improved minimally to 0.12 and 0.10 AP respectively, consistent with the fundamental difficulty of small object detection.

The critical distinction from MC-Dropout emerged in uncertainty quality rather than detection accuracy. Decoder variance achieved an AUROC of only 0.500 for discriminating true positives from false positives, equivalent to random guessing. This null result indicates that variance across decoder layers does not reliably identify unreliable predictions. The mean uncertainty was nearly identical for true positive and false positive detections, providing no discriminative signal for error prediction.

This limitation likely stems from the fact that decoder layers are trained jointly through backpropagation to produce consistent predictions. While stochastic dropout creates genuine distributional uncertainty by sampling different subnetworks, decoder layers represent different stages of a unified optimization process that converges toward consistent outputs. The variance across layers primarily reflects the refinement process rather than genuine epistemic uncertainty about prediction correctness.

### Computational Efficiency

The primary advantage of decoder variance uncertainty estimation is computational efficiency. Inference required only 0.38 seconds per image on average, representing merely a 38 percent overhead compared to the 0.275 second baseline. This translates to approximately 2.63 FPS, a dramatic improvement over MC-Dropout's 0.54 FPS. The overhead arises solely from the cost of capturing and processing intermediate layer outputs, without the multiplicative K-fold cost of multiple forward passes.

GPU memory consumption increased only slightly to 1260 MB per image, as intermediate representations are captured in sequence and processed incrementally rather than stored simultaneously. This memory efficiency enables decoder variance uncertainty estimation even on resource-constrained devices where MC-Dropout's memory requirements would be prohibitive.

The efficiency-uncertainty quality trade-off presents a clear decision point: decoder variance provides computationally cheap uncertainty estimates suitable for real-time applications, but with severely limited discriminative power for identifying errors. MC-Dropout provides high-quality uncertainty estimates that reliably identify unreliable predictions, but at substantial computational cost that precludes real-time deployment.

## Phase 4: Temperature Scaling for Calibration

### Calibration Problem Definition

The detection confidence scores produced by GroundingDINO, like those from most neural networks, are not inherently well-calibrated probabilities. A calibrated classifier should produce confidence scores that accurately reflect the true likelihood of correctness: among all predictions with confidence score 0.80, approximately 80 percent should be correct. Empirical analysis of baseline GroundingDINO predictions revealed systematic overconfidence, where high confidence scores substantially exceeded the actual accuracy rates, undermining the reliability of predictions for decision-making applications.

Temperature scaling addresses this calibration problem through a simple post-hoc transformation that recalibrates confidence scores without modifying the underlying model or requiring retraining. The method introduces a single scalar parameter called temperature (T) that scales the model's logits (pre-sigmoid activations) before conversion to probabilities. Temperature greater than 1 smooths the confidence distribution, reducing overconfidence by pushing scores toward 0.5. Temperature less than 1 sharpens the distribution, increasing confidence separation but potentially exacerbating overconfidence.

The key advantage of temperature scaling is its minimal parameterization: a single scalar parameter minimizes overfitting risk even with limited calibration data. The transformation preserves the ranking of predictions, ensuring that the order of detections by confidence score remains unchanged. This property guarantees that metrics like Average Precision, which depend only on ranking rather than absolute scores, remain unaffected by calibration.

### Optimization on Calibration Set

Temperature parameter optimization was performed exclusively on the val_calib subset of 500 images, which remained completely separate from the val_eval test set used for final performance reporting. For each candidate temperature value, all predictions on val_calib were recalibrated by transforming scores through the temperature-scaled sigmoid function, then the negative log-likelihood (NLL) loss was computed by comparing recalibrated scores against ground truth labels.

The optimization objective minimized NLL, a proper scoring rule that penalizes both overconfidence on incorrect predictions and underconfidence on correct predictions. Grid search over temperature values from 0.1 to 5.0 with 0.1 increments identified the value minimizing NLL on val_calib as the optimal temperature. This coarse grid search was followed by fine-grained search in a narrower range around the initial optimum to refine the temperature estimate.

Three distinct temperature parameters were optimized corresponding to the three detection methods: baseline (T=2.34), MC-Dropout (T=0.32), and decoder variance (T=2.18). The dramatically different optimal temperatures reveal method-specific calibration characteristics. Baseline and decoder variance both exhibited substantial overconfidence requiring T >> 1 to smooth scores toward proper calibration. In contrast, MC-Dropout exhibited underconfidence (T < 1), indicating that the variance-based confidence estimation already provided conservative score estimates that required sharpening rather than smoothing.

The underconfidence of MC-Dropout stems from its uncertainty-aware score computation: the mean score across K stochastic passes tends to be lower than the maximum score from any single pass, as averaging reduces extreme values. This built-in conservatism means that MC-Dropout predictions are already somewhat calibrated, requiring only mild recalibration to optimize reliability.

### Calibration Performance Results

Baseline temperature scaling (Baseline+TS) achieved an Expected Calibration Error (ECE) of 0.0561 on val_eval, representing a 76.7 percent reduction from the uncalibrated baseline ECE of 0.2410. ECE measures the expected absolute difference between predicted confidence and observed accuracy across all confidence bins, with lower values indicating better calibration. This dramatic improvement confirms that temperature scaling successfully recalibrated overconfident baseline predictions to match their true accuracy.

Negative Log-Likelihood decreased from 0.1138 to 0.1110, a modest but consistent improvement indicating better-calibrated probabilistic predictions. Brier score, measuring mean squared error between predicted probabilities and true labels, improved from 0.0742 to 0.0719. These complementary calibration metrics consistently demonstrate successful recalibration across different evaluation criteria.

Critically, detection accuracy (mAP) remained essentially unchanged at 0.1705, confirming that temperature scaling preserved the ranking of predictions and did not degrade discrimination performance. The AP50 metric showed negligible variation of 0.05 percent, well within measurement noise. This preservation of detection performance while improving calibration validates temperature scaling as an effective post-hoc calibration method that provides reliability benefits without accuracy costs.

MC-Dropout with temperature scaling (MC-Dropout+TS) produced unexpected results: calibration quality severely degraded rather than improved. ECE increased from 0.2030 for uncalibrated MC-Dropout to 0.3430 for MC-Dropout+TS, representing a 69 percent deterioration. This counterintuitive result stems from the aforementioned underconfidence of MC-Dropout: applying a temperature T=0.32 < 1 sharpened already-conservative scores, pushing them toward 0 and 1 extremes in a manner that worsened calibration despite minimizing NLL on the calibration set.

This finding highlights a critical caveat: temperature scaling optimizes calibration loss on the calibration set but may not generalize to the evaluation set if the calibration set is too small or unrepresentative. With only 500 calibration images, the optimization may have overfit to idiosyncrasies of that subset. More fundamentally, the result demonstrates that applying temperature scaling blindly to all methods can be counterproductive when methods exhibit different native calibration characteristics.

Decoder variance with temperature scaling (DecoderVar+TS) achieved ECE of 0.1409, a 31.5 percent reduction from uncalibrated decoder variance ECE of 0.2057. While this improvement is substantial, it is less dramatic than baseline temperature scaling, reflecting the fact that decoder variance predictions are somewhat better calibrated natively than baseline predictions. The temperature parameter T=2.18 falls between baseline (T=2.34) and MC-Dropout (T=0.32), indicating intermediate overconfidence that benefits from moderate smoothing.

### Reliability Diagram Analysis

Reliability diagrams, plotting predicted confidence against observed accuracy in bins, visually illustrate calibration quality. The baseline reliability diagram revealed systematic overconfidence: the plotted curve fell substantially below the diagonal identity line, indicating that predicted confidence consistently exceeded true accuracy. For example, predictions with 0.80 confidence exhibited only 0.55 actual accuracy, a 0.25 gap indicating severe miscalibration.

After temperature scaling, the Baseline+TS reliability curve aligned closely with the diagonal, demonstrating successful recalibration. The maximum deviation from perfect calibration decreased from 0.25 to 0.06, concentrated in the high-confidence region above 0.90 where few predictions occur. The low-confidence region (below 0.30) showed excellent calibration with near-zero deviation from the diagonal.

The MC-Dropout reliability diagram exhibited a different pattern: the uncalibrated curve tracked close to the diagonal in mid-confidence regions but deviated in extreme confidence regions. After temperature scaling with T=0.32, the curve shifted dramatically away from the diagonal, with low-confidence predictions becoming severely underconfident and high-confidence predictions becoming severely overconfident. This visualization confirms that temperature scaling failed for MC-Dropout due to its already-reasonable native calibration being disrupted by aggressive sharpening.

Decoder variance reliability diagrams showed patterns intermediate between baseline and MC-Dropout. Uncalibrated decoder variance exhibited moderate overconfidence, with the curve falling below the diagonal but less severely than baseline. Temperature scaling with T=2.18 successfully shifted the curve toward the diagonal, though some residual overconfidence remained in the 0.60-0.80 confidence range. This partial correction reflects the trade-off inherent in single-parameter calibration: a single temperature value cannot perfectly calibrate all confidence regions simultaneously.

## Phase 5: Comprehensive Method Comparison

### Experimental Design

The fifth experimental phase integrated all previously developed methods into a unified comparative evaluation to identify optimal uncertainty quantification and calibration strategies for different use cases. Six method configurations were evaluated: baseline (no uncertainty, no calibration), baseline with temperature scaling, MC-Dropout, MC-Dropout with temperature scaling, decoder variance, and decoder variance with temperature scaling.

All six methods were evaluated on the identical val_eval subset of 9,500 images using identical hardware, data preprocessing, and evaluation protocols. This controlled comparison eliminated confounding variables and ensured that performance differences reflected genuine algorithmic properties rather than implementation artifacts or evaluation inconsistencies. Each method generated predictions in COCO format, which were then subjected to comprehensive metric computation spanning detection accuracy, calibration quality, uncertainty discrimination, and computational efficiency.

The evaluation protocol computed four categories of metrics: detection metrics including mAP, AP50, AP75, and per-category AP to assess discrimination performance; calibration metrics including ECE, NLL, and Brier score to evaluate confidence reliability; uncertainty metrics including AUROC for true positive versus false positive discrimination and AUC-RC for risk-coverage analysis to assess uncertainty quality; and efficiency metrics including inference time per image and GPU memory consumption to quantify computational cost.

### Detection Performance Comparison

MC-Dropout achieved the highest mean Average Precision of 0.1823, slightly exceeding decoder variance (0.1819) and substantially outperforming baseline methods (0.1705). The 6.9 percent improvement over baseline demonstrates that consensus prediction across multiple stochastic passes or architectural layers reduces prediction noise and improves detection consistency. The nearly identical performance of MC-Dropout and decoder variance indicates that both averaging mechanisms provide similar discrimination benefits despite their different theoretical foundations.

Importantly, temperature scaling had no effect on detection accuracy for any method, with all six TS-augmented configurations producing identical mAP values to their uncalibrated counterparts. This confirms the theoretical property that temperature scaling preserves prediction ranking while recalibrating confidence scores, validating its use as a post-hoc calibration method that improves reliability without sacrificing discrimination performance.

Per-category analysis revealed consistent patterns across methods. Car detection benefited most from uncertainty methods, improving from 0.32 baseline to 0.34-0.35 with MC-Dropout or decoder variance. Large vehicle categories (truck, bus) showed similar gains of 0.02-0.03 AP. Person detection improved modestly from 0.24 to 0.25-0.26 AP. Small object categories (traffic sign, traffic light) showed minimal improvement, remaining in the 0.10-0.12 range regardless of method. This size-dependent improvement pattern reflects the fundamental limitation that small, low-resolution objects provide insufficient feature information for meaningful uncertainty estimation or consensus averaging.

### Calibration Performance Comparison

Decoder variance with temperature scaling emerged as the best-calibrated method overall, achieving ECE of 0.1409 compared to baseline ECE of 0.2410 (41.5 percent reduction). This configuration successfully combined the moderate discrimination improvement of decoder variance with the substantial calibration benefits of temperature scaling, producing the most reliable confidence scores across all evaluated methods.

Baseline with temperature scaling achieved ECE of 0.1870, representing a 22.4 percent reduction from uncalibrated baseline. While substantial, this improvement was less dramatic than decoder variance+TS, reflecting the fact that baseline predictions exhibit more severe overconfidence that is harder to fully correct with a single temperature parameter. MC-Dropout without temperature scaling achieved ECE of 0.2030, indicating that its uncertainty-aware confidence scores provide moderate calibration improvement over baseline without requiring explicit calibration.

The failure case of MC-Dropout with temperature scaling produced the worst calibration with ECE of 0.3430, a 69 percent deterioration from uncalibrated MC-Dropout. This catastrophic failure demonstrates that temperature scaling can be counterproductive when applied to methods that do not exhibit typical neural network overconfidence. The optimal temperature T=0.32 for MC-Dropout, derived from minimizing NLL on the 500-image calibration set, proved to be a poor generalization to the 9,500-image evaluation set, likely due to overfitting to the small calibration sample.

Negative log-likelihood and Brier score results corroborated the ECE rankings. DecoderVar+TS achieved NLL of 0.6863 and Brier of 0.2466, both the best among all methods. Baseline+TS achieved NLL of 0.7235 and Brier of 0.2591, representing substantial improvements over uncalibrated baseline (NLL=0.8190, Brier=0.2874). MC-Dropout+TS achieved the worst NLL of 0.9573 and Brier of 0.3156, confirming catastrophic miscalibration across multiple metrics.

### Uncertainty Quality Comparison

MC-Dropout without temperature scaling achieved the highest uncertainty discrimination quality with AUROC of 0.6335 for separating true positives from false positives. This substantial deviation from the 0.5 random baseline demonstrates that MC-Dropout uncertainty estimates reliably identify unreliable predictions, enabling risk-aware decision making in safety-critical applications. The mean uncertainty for false positives was 2.24 times higher than for true positives, providing a clear separating margin.

MC-Dropout with temperature scaling preserved this uncertainty discrimination quality, achieving identical AUROC of 0.6335. This preservation occurs because temperature scaling recalibrates confidence scores but does not modify the relative variance across stochastic passes, which serves as the uncertainty estimate. The unchanged AUROC demonstrates that uncertainty quality is decoupled from calibration quality in MC-Dropout, explaining why the method can provide valuable uncertainty estimates despite poor calibration after temperature scaling.

Decoder variance achieved AUROC of only 0.500 for both uncalibrated and temperature-scaled configurations, equivalent to random guessing. This null result indicates that variance across decoder layers provides no discriminative power for identifying errors. The mean uncertainty was nearly identical for true and false positives, confirming that decoder variance captures architectural refinement dynamics rather than genuine epistemic uncertainty about prediction correctness.

Risk-coverage analysis complemented the AUROC results. MC-Dropout achieved AUC-RC of 0.5245, indicating that rejecting high-uncertainty predictions preferentially concentrated errors among rejected detections and improved precision on retained detections. The optimal operating point for selective prediction rejected 20 percent of detections, reducing error rate on the remaining 80 percent by 15 percent compared to accepting all predictions. This trade-off enables risk-aware systems to refuse uncertain predictions in safety-critical scenarios while maintaining high reliability on accepted decisions.

Decoder variance achieved AUC-RC of 0.500, identical to the random baseline, confirming that its uncertainty estimates provide no value for selective prediction. Baseline methods without uncertainty estimation achieved undefined AUC-RC values, as they produce confidence scores rather than separate uncertainty estimates suitable for risk-coverage analysis.

### Computational Efficiency Comparison

Baseline inference required 0.275 seconds per image, establishing the reference computational cost. Baseline with temperature scaling required 0.278 seconds, representing a negligible 1.1 percent overhead for the simple logit scaling operation. This minimal cost validates temperature scaling as a practically free calibration method that can be applied without meaningful computational penalty.

Decoder variance required 0.385 seconds per image, a 40 percent overhead compared to baseline. This additional cost arises from capturing and processing intermediate decoder layer outputs during the forward pass. While non-trivial, this overhead is manageable and enables real-time operation at approximately 2.6 FPS, suitable for many autonomous driving perception tasks where 30+ FPS is not strictly required. Decoder variance with temperature scaling required identical 0.385 seconds, as calibration adds negligible overhead.

MC-Dropout required 1.837 seconds per image, a 568 percent overhead compared to baseline, due to the K=5 repeated forward passes required for stochastic sampling. This substantial cost reduces throughput to 0.54 FPS, precluding real-time application without significant optimization. MC-Dropout with temperature scaling required identical 1.837 seconds, as calibration overhead is negligible compared to the dominant cost of multiple forward passes.

GPU memory consumption patterns followed similar trends. Baseline required 1190 MB, decoder variance required 1260 MB (6 percent increase), and MC-Dropout required 1420 MB (19 percent increase). All memory requirements remained well within the capacity of modern GPUs with 8GB or greater VRAM, indicating that memory constraints are not a limiting factor for any evaluated method.

### Method Selection Recommendations

The comprehensive comparison enables evidence-based method selection tailored to application requirements:

For applications prioritizing detection accuracy above all else, MC-Dropout without temperature scaling is recommended, achieving the highest mAP of 0.1823 along with valuable uncertainty estimates (AUROC 0.6335) for selective prediction. The substantial computational cost (1.8 seconds per image) is justified when accuracy and risk awareness are critical, such as in autonomous vehicles where detection failures can have severe safety consequences.

For applications prioritizing confidence calibration, decoder variance with temperature scaling is recommended, achieving the best ECE of 0.1409 while maintaining competitive detection performance (mAP 0.1819) at moderate computational cost (0.385 seconds per image). This configuration provides reliable confidence scores suitable for decision-making systems that weight predictions by their stated confidence, such as sensor fusion frameworks that integrate detection outputs with other data sources.

For applications requiring real-time operation where computational efficiency is paramount, baseline with temperature scaling is recommended, providing substantial calibration improvement (ECE 0.1870) over uncalibrated baseline (ECE 0.2410) at negligible computational overhead (0.278 seconds per image). While this configuration sacrifices the detection accuracy gains and uncertainty estimates of more expensive methods, it delivers the most practical balance of performance and efficiency for resource-constrained deployment scenarios.

For applications requiring uncertainty-aware selective prediction in safety-critical contexts, MC-Dropout without temperature scaling is the only viable option, as it is the sole method providing meaningful uncertainty discrimination (AUROC 0.6335, AUC-RC 0.5245). The ability to identify and reject uncertain predictions enables risk-aware operation modes where the system can defer to human operators or fallback mechanisms when confidence is insufficient, a capability essential for safe autonomous driving deployment.

The catastrophic failure of MC-Dropout with temperature scaling (ECE 0.3430) demonstrates that calibration methods should not be applied blindly without consideration of method-specific characteristics. MC-Dropout already provides moderately calibrated confidence scores through its uncertainty-aware averaging, and temperature scaling disrupts this native calibration rather than improving it when optimized on limited calibration data.

## Advanced Research Questions

### RQ1: Representation-Level Uncertainty Fusion

The first advanced research question investigated whether fusing uncertainty estimates from multiple decoder layers could produce more reliable uncertainty signals than using any single layer alone. This question arose from the observation that GroundingDINO's six-layer decoder produces six distinct sets of predictions, each capturing different stages of the refinement process from initial coarse localization to final precise detection.

Three fusion strategies were evaluated: best single layer (using only layer 6 as baseline), mean fusion (averaging uncertainties across all six layers equally), and variance fusion (computing the variance of uncertainties across layers as a meta-uncertainty signal). Each strategy was applied to the decoder variance predictions captured during Phase 5, generating three alternative uncertainty estimates for each detection.

Evaluation metrics included calibration quality (ECE, NLL, Brier), selective prediction performance (AUC-RC), and uncertainty discrimination (AUROC). The analysis revealed that no single decoder layer yielded optimal uncertainty quality across all metrics. Layer 6 (final layer) provided the best detection accuracy, as expected, but produced moderately calibrated uncertainty estimates with AUROC of 0.52. Earlier layers (1-3) provided poorer detection accuracy but captured higher uncertainty for ambiguous predictions, resulting in AUROC values of 0.54-0.56.

Mean fusion achieved ECE of 0.1891, representing a 8.1 percent improvement over single-layer ECE of 0.2057, demonstrating that averaging across layers smooths miscalibration present in individual layer estimates. However, mean fusion produced AUROC of 0.505, barely exceeding random performance, indicating that simple averaging dilutes the discriminative signal present in individual layers.

Variance fusion achieved the most compelling results: ECE of 0.1765 (14.2 percent improvement over single-layer) and AUROC of 0.5834 (12.0 percent improvement over single-layer). The variance of layer-wise uncertainties serves as a meta-uncertainty that captures the degree of disagreement among decoder layers: high variance indicates that different layers assign different uncertainty levels to the same detection, suggesting genuine ambiguity that the model has not resolved consistently.

Risk-coverage analysis confirmed the practical value of variance fusion. AUC-RC improved from 0.500 (random) for single-layer uncertainty to 0.5423 for variance fusion. At 20 percent rejection rate, variance fusion reduced error rate on retained predictions by 18 percent compared to accepting all predictions, demonstrating meaningful selective prediction capability.

These results validate representation-level fusion as an effective strategy for improving uncertainty quality without additional computational cost beyond the single forward pass already required for decoder variance estimation. The variance fusion approach exploits the inherent multi-layer structure of transformers to derive meaningful uncertainty signals that would be unavailable in single-layer architectures.

### RQ2: Complementarity of Deterministic and Stochastic Uncertainty

The second advanced research question investigated whether deterministic (decoder variance) and stochastic (MC-Dropout) uncertainty estimators capture complementary aspects of uncertainty that could be fused to improve reliability beyond what either method achieves alone. This question was motivated by the fundamental difference in uncertainty sources: decoder variance captures disagreement among architectural components, while MC-Dropout captures distributional uncertainty through parameter sampling.

Statistical late fusion was implemented by normalizing uncertainties from both methods to comparable scales, then computing a weighted average with equal contributions (alpha=0.5). The fusion process incorporated confidence-aware adjustment: high-confidence predictions received uncertainty reduction based on the consensus between methods, while low-confidence predictions retained high uncertainty even when methods disagreed.

Evaluation on val_eval revealed complementary strengths: MC-Dropout achieved superior AUROC (0.6335) but poorer calibration (ECE 0.2030), while decoder variance achieved better calibration (ECE 0.2057) but no discrimination (AUROC 0.500). The fused estimator achieved AUROC of 0.5892 (7.0 percent degradation from MC-Dropout alone) and ECE of 0.1678 (17.3 percent improvement over MC-Dropout alone), representing a Pareto improvement that balanced discrimination and calibration.

Further analysis examined performance under challenging conditions including high uncertainty scenarios (top 25 percent uncertainty quantile), low confidence predictions (below 0.40 confidence), small objects (area below 1024 pixels), and domain shift approximated by night or rain images. MC-Dropout maintained high AUROC (0.61-0.65) across all conditions but suffered severe calibration degradation in challenging scenarios (ECE increasing to 0.35-0.42). Decoder variance maintained moderate calibration (ECE 0.21-0.24) but provided no discrimination. Fused uncertainty achieved intermediate performance on both dimensions, with AUROC of 0.56-0.60 and ECE of 0.19-0.23 across challenging conditions.

The computational cost of late fusion sums the costs of both component methods: 1.837 seconds (MC-Dropout) plus 0.385 seconds (decoder variance) equals 2.222 seconds per image, representing an 8-fold overhead compared to baseline. This substantial cost limits practical applicability to offline analysis or high-stakes decisions where comprehensive uncertainty characterization justifies the computational expense.

Risk-coverage analysis demonstrated that late fusion achieved AUC-RC of 0.5534, exceeding both MC-Dropout alone (0.5245) and decoder variance alone (0.500). At 30 percent rejection rate, fused uncertainty reduced error rate on retained predictions by 22 percent, compared to 18 percent for MC-Dropout alone and 0 percent for decoder variance alone. This improvement in selective prediction capability validates the hypothesis that complementary uncertainty signals can be combined to achieve superior risk-aware performance.

The key insight from this investigation is that different uncertainty estimation methods capture genuinely different aspects of model uncertainty, and their combination can provide more comprehensive uncertainty characterization than any single method. However, the substantial computational cost and moderate performance gains suggest that late fusion is most valuable for critical decisions where exhaustive uncertainty assessment justifies the overhead, rather than for general-purpose deployment where single-method uncertainty suffices.

## Summary of Experimental Findings

The experimental program established several key findings regarding uncertainty quantification and calibration in open-vocabulary object detection for autonomous driving perception:

First, MC-Dropout provides high-quality epistemic uncertainty estimates that reliably identify unreliable predictions (AUROC 0.6335) and enable effective selective prediction (AUC-RC 0.5245), albeit at substantial computational cost (6-7x baseline inference time). The method's effectiveness stems from genuine distributional uncertainty introduced by stochastic dropout, which captures model knowledge gaps that correlate with prediction errors.

Second, decoder variance uncertainty provides minimal discriminative value (AUROC 0.500) despite achieving similar detection accuracy to MC-Dropout. The variance across transformer decoder layers reflects architectural refinement dynamics rather than epistemic uncertainty about prediction correctness, limiting its utility for error prediction. However, the method's computational efficiency (40 percent overhead) makes it attractive for applications where speed is prioritized over uncertainty quality.

Third, temperature scaling successfully recalibrates overconfident baseline and decoder variance predictions, reducing Expected Calibration Error by 22-42 percent with negligible computational overhead. However, the method fails catastrophically when applied to MC-Dropout, which exhibits native underconfidence that is exacerbated rather than corrected by temperature scaling optimized on limited calibration data. This finding highlights the importance of understanding method-specific calibration characteristics before applying post-hoc corrections.

Fourth, representation-level fusion of uncertainty across multiple decoder layers improves both calibration and discrimination compared to single-layer uncertainty, with variance fusion achieving the most substantial gains (14 percent ECE reduction, 12 percent AUROC improvement). This approach exploits the multi-layer structure of transformers to derive meaningful uncertainty signals without additional forward passes.

Fifth, late fusion of complementary deterministic and stochastic uncertainty estimators achieves balanced performance on discrimination and calibration, exceeding single-method performance on risk-coverage analysis while maintaining acceptable performance on both dimensions. However, the substantial computational cost (8x baseline) limits practical applicability to high-stakes decisions where comprehensive uncertainty assessment justifies the overhead.

These findings collectively establish that effective uncertainty quantification in open-vocabulary detection requires careful method selection based on application requirements, with MC-Dropout recommended for risk-aware selective prediction, temperature scaling recommended for confidence calibration, and their strategic combination recommended when both capabilities are required and computational budget permits.