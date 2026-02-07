# Network Architecture

## Overview

This research employs GroundingDINO, a state-of-the-art open-vocabulary object detection model, as the foundational architecture for investigating epistemic uncertainty quantification in autonomous driving perception systems. GroundingDINO represents a paradigm shift from traditional closed-set object detectors by leveraging vision-language fusion through transformer-based architectures, enabling the detection of arbitrary object categories described in natural language without requiring explicit training on those categories.

The selection of GroundingDINO for this work is motivated by three key considerations: first, its open-vocabulary capability addresses the long-tail distribution of objects encountered in real-world driving scenarios where pre-defined category sets are insufficient; second, its transformer-based architecture provides multiple intermediate representations suitable for uncertainty quantification; and third, its competitive performance on standard benchmarks ensures that uncertainty estimates are derived from a robust detection backbone.

## Architectural Components

### Vision Backbone: Swin Transformer

The visual encoding component of GroundingDINO utilizes the Swin Transformer architecture, a hierarchical vision transformer that processes images through a sequence of shifted window-based self-attention operations. Unlike traditional convolutional neural networks that rely on spatial locality through fixed receptive fields, Swin Transformer constructs hierarchical feature representations through increasingly larger spatial windows across multiple stages.

The Swin Transformer backbone in GroundingDINO operates in four stages with progressively increasing patch sizes and decreasing spatial resolutions, mirroring the hierarchical structure of convolutional networks while maintaining the global modeling capacity of transformers. The input image is initially partitioned into non-overlapping patches of 4×4 pixels, with each patch treated as a token. These patch tokens are then linearly embedded into a high-dimensional feature space where subsequent transformer layers operate.

Each Swin Transformer stage consists of multiple transformer blocks with shifted window-based multi-head self-attention mechanisms. The window-based approach partitions the feature map into fixed-size windows and computes self-attention within each window independently, significantly reducing computational complexity from quadratic to linear with respect to image size. The shifting operation between consecutive layers enables cross-window connections, allowing information to flow across different spatial regions.

The hierarchical nature of Swin Transformer produces multi-scale feature representations at four different resolutions, which are subsequently fed into the fusion transformer for cross-modal alignment. This multi-scale design is particularly important for object detection tasks as it enables the detection of objects at various sizes, from small traffic signs to large vehicles, within the same framework.

### Language Encoder: BERT

The textual encoding component employs BERT, a bidirectional transformer-based language model pre-trained on large-scale text corpora. BERT processes the input text prompts, which specify the object categories of interest, and generates contextualized token embeddings that capture semantic relationships between words.

For this research, text prompts consist of concatenated object category names relevant to autonomous driving scenarios, including "person", "car", "truck", "bus", "bicycle", "motorcycle", "traffic light", "traffic sign", "rider", and "train". The BERT encoder tokenizes these prompts, adds special classification and separator tokens, and processes the sequence through multiple transformer layers to produce contextual embeddings for each token.

The BERT encoder generates embeddings that encode not only the individual semantic meaning of each category name but also the relationships between different categories through its bidirectional attention mechanism. These contextualized embeddings serve as queries for the subsequent cross-modal fusion process, enabling the model to ground textual descriptions in visual features.

### Cross-Modal Fusion: Transformer Encoder-Decoder

The core innovation of GroundingDINO lies in its transformer-based fusion mechanism that bridges the visual and linguistic modalities through cross-attention operations. This fusion architecture consists of an encoder-decoder transformer structure where visual features from the Swin backbone and textual features from BERT are integrated.

The transformer encoder processes both visual patch tokens and textual word tokens jointly through self-attention mechanisms, allowing bidirectional information flow between modalities. This early fusion enables the model to establish correspondences between visual regions and textual descriptions at multiple levels of abstraction. The encoder applies multiple layers of multi-head self-attention, where attention weights are computed across all visual and textual tokens simultaneously.

Following the encoder, the transformer decoder generates object queries through a learned set of query embeddings. These queries interact with the fused visual-linguistic features through cross-attention mechanisms, where each query attends to relevant regions in the feature space. The decoder architecture consists of six sequential layers, with each layer refining the object representations through three sub-modules: self-attention among queries, cross-attention to encoder outputs, and feed-forward networks.

The cross-attention mechanism in the decoder is particularly important for grounding: each object query learns to attend to visual regions that correspond to the textual descriptions while suppressing irrelevant regions. This attention-based grounding enables the model to localize objects based on their semantic descriptions rather than fixed category indices.

Each decoder layer produces intermediate object embeddings that represent the model's progressively refined understanding of object locations and categories. The variance in these intermediate representations across layers serves as a natural uncertainty signal, as consistent representations across layers indicate confident predictions while divergent representations suggest uncertainty.

### Detection Heads

The final components of the architecture are the detection heads, which transform the decoder output embeddings into explicit object predictions. GroundingDINO employs two parallel prediction heads: a classification head and a bounding box regression head.

The classification head consists of a linear projection layer that maps each object query embedding to class logits. These logits represent the similarity between the query embedding and the textual embeddings of each category, implemented through dot-product attention. The sigmoid activation is then applied to produce independent probability scores for each category, enabling multi-label predictions when objects match multiple textual descriptions.

The bounding box regression head comprises a three-layer multilayer perceptron that predicts normalized box coordinates in the format center x, center y, width, height relative to the image dimensions. The regression head is designed to be category-agnostic, predicting spatial locations independently of the classification scores.

A critical architectural detail relevant to this research is that both detection heads in GroundingDINO are deterministic linear or MLP layers without dropout modules. This design choice, while standard in modern object detectors for prediction stability, has important implications for uncertainty quantification methods that rely on stochastic forward passes.

## Dropout Module Distribution

Understanding the placement and characteristics of dropout modules within GroundingDINO is essential for implementing Monte Carlo Dropout-based uncertainty estimation. Through systematic architectural inspection, this research identified 37 dropout modules distributed across the model, all located within the transformer encoder and decoder layers rather than the detection heads.

Each transformer layer in both the encoder and decoder contains dropout operations applied after the attention mechanism and within the feed-forward network. These dropout modules operate with a probability p=0.1, meaning that during training, 10 percent of neurons are randomly deactivated in each forward pass. This relatively conservative dropout rate reflects the transformer architecture's inherent capacity for regularization through attention mechanisms.

The absence of dropout in the final detection heads is a deliberate architectural decision that prioritizes prediction stability and deterministic behavior during inference. While this design improves the consistency of predictions, it presents a challenge for traditional Monte Carlo Dropout implementations that assume stochastic behavior in the prediction layers.

In this research, the 37 dropout modules in the encoder-decoder layers are leveraged for uncertainty estimation by maintaining them in training mode during inference while keeping other components, such as batch normalization layers, in evaluation mode. This selective activation enables stochastic forward passes where dropout masks vary across multiple runs, producing the prediction variance necessary for epistemic uncertainty quantification.

The distribution of dropout across multiple transformer layers rather than just the prediction heads has a specific implication: the uncertainty captured by this approach primarily reflects ambiguity in the feature representations rather than prediction-level stochasticity. This means that high variance across stochastic passes indicates that the model's intermediate representations are unstable, suggesting uncertainty about which visual features correspond to the textual queries.

## Architectural Considerations for Uncertainty Quantification

The transformer-based architecture of GroundingDINO introduces both opportunities and challenges for uncertainty quantification compared to traditional convolutional object detectors. Understanding these architectural characteristics is crucial for designing effective uncertainty estimation methods.

### Multi-Layer Representations

Unlike convolutional architectures that produce a single feature map per stage, the transformer decoder in GroundingDINO generates object query embeddings at each of its six layers. Each layer progressively refines the object representations through self-attention and cross-attention operations. This multi-layer structure provides a natural mechanism for uncertainty estimation: if the model is confident about a detection, the embeddings should converge to consistent representations across layers; conversely, high variance across layers suggests that the model has not reached a stable interpretation.

This architectural property enables decoder variance-based uncertainty estimation, which measures the consistency of object representations across decoder layers. By computing the variance of embedding magnitudes or attention patterns across layers, the model can derive uncertainty estimates from its internal processing without requiring multiple stochastic forward passes.

### Cross-Modal Attention Patterns

The vision-language fusion in GroundingDINO introduces an additional source of uncertainty not present in traditional single-modality detectors. Uncertainty can arise from ambiguous visual features, ambiguous textual descriptions, or misalignment between the two modalities. For instance, a small, distant object might produce weak visual features that match multiple textual descriptions, resulting in high uncertainty.

The cross-attention mechanisms in the decoder provide insight into this multimodal uncertainty through attention weight distributions. Sharp attention distributions, where a query strongly attends to a small set of visual features, indicate confident grounding. Diffuse attention distributions, where attention is spread across many regions, suggest uncertainty about the object's location or category.

### Deterministic Prediction Heads

The deterministic nature of the detection heads in GroundingDINO constrains the types of uncertainty that can be directly estimated from the prediction layer. Unlike Bayesian neural networks or probabilistic detectors that model prediction distributions explicitly, GroundingDINO produces point estimates for class scores and bounding boxes.

This architectural characteristic motivates the exploration of uncertainty quantification methods that operate on intermediate representations rather than final predictions. Monte Carlo Dropout captures uncertainty in the feature space through stochastic encoder-decoder activations, while temperature scaling recalibrates the predicted confidence scores post-hoc without modifying the prediction mechanism itself.

### Transformer-Specific Regularization

Transformers employ several forms of regularization beyond dropout, including layer normalization, attention dropout, and path dropout in some variants. These mechanisms interact with dropout-based uncertainty estimation in complex ways. Layer normalization, applied before each attention and feed-forward sub-layer, stabilizes training but also reduces the magnitude of uncertainty signals by normalizing intermediate activations.

Attention dropout, when present, applies dropout to attention weights rather than hidden states, creating uncertainty in which features are attended to rather than in the features themselves. This form of dropout can be particularly informative for uncertainty estimation as it directly reflects the model's uncertainty about feature relevance.

## Architectural Comparison with Traditional Detectors

To contextualize the architectural choices in GroundingDINO and their implications for uncertainty quantification, it is instructive to compare this transformer-based approach with traditional convolutional object detectors.

Traditional detectors such as YOLO, Faster R-CNN, and RetinaNet typically employ convolutional backbones like ResNet or EfficientNet for feature extraction, followed by specialized detection heads that predict class probabilities and bounding box coordinates from feature maps at multiple scales. These architectures often incorporate dropout in the classification and regression heads, making Monte Carlo Dropout straightforward to implement by simply keeping dropout active during inference.

In contrast, GroundingDINO's transformer architecture processes images as sequences of patches rather than feature maps, computes global self-attention rather than local convolutions, and fuses visual and textual information through cross-attention rather than single-modality processing. These architectural differences necessitate adaptations to traditional uncertainty quantification approaches.

The attention-based nature of transformers provides richer intermediate representations for uncertainty estimation. While convolutional features are primarily characterized by their spatial activations, transformer embeddings encode relational information through attention patterns. This enables uncertainty quantification methods like decoder variance that exploit the multi-layer refinement process inherent to transformers.

Furthermore, the open-vocabulary capability of GroundingDINO introduces a categorically different uncertainty profile compared to closed-set detectors. Traditional detectors exhibit uncertainty primarily in distinguishing between fixed categories and localizing objects accurately. GroundingDINO additionally faces uncertainty in grounding arbitrary textual descriptions to visual features, introducing linguistic ambiguity as a source of epistemic uncertainty.

## Implementation Details

The implementation of GroundingDINO used in this research employs the Swin-Tiny (SwinT-OGC) configuration with the following specifications: the Swin Transformer backbone has approximately 28 million parameters organized into four stages with depths of 2, 2, 6, and 2 transformer blocks respectively. The embedding dimension is 96 in the first stage and increases by a factor of two at each subsequent stage through patch merging operations.

The transformer encoder-decoder fusion architecture consists of 6 encoder layers and 6 decoder layers, each with 8 attention heads operating on embeddings of dimension 256. The feed-forward networks within each layer expand the dimension by a factor of four, applying activation functions and dropout before projecting back to the original dimension.

The BERT text encoder employs 6 transformer layers with 512-dimensional embeddings, sufficient for encoding the relatively simple text prompts used in object detection tasks. The language model was initialized with weights pre-trained on standard BERT corpora and then fine-tuned during GroundingDINO training on detection datasets with caption annotations.

Model inference processes images at their native resolution without resizing, taking advantage of the Swin Transformer's efficient handling of variable input sizes through window-based attention. For the BDD100K dataset used in this research, images have a resolution of 1280×720 pixels, resulting in feature maps of various sizes across the hierarchical stages.

The detection heads produce a maximum of 900 object queries per image, with non-maximum suppression applied post-processing to remove redundant detections. The classification head generates scores in the range zero to one through sigmoid activation, while the bounding box head produces normalized coordinates that are scaled to pixel coordinates during post-processing.

All experiments were conducted using the pre-trained GroundingDINO weights released by the original authors, trained on a combination of object detection datasets including COCO, Objects365, and Visual Genome, along with caption datasets for vision-language grounding. No fine-tuning was performed on the BDD100K dataset, ensuring that the uncertainty estimates reflect the model's epistemic uncertainty on out-of-distribution driving scenarios.

## Architectural Implications for Research Questions

The architectural properties of GroundingDINO directly inform the research questions investigated in this work and the methods developed to address them.

The first research question, concerning the quantification of epistemic uncertainty in open-vocabulary detection, is enabled by the transformer architecture's multiple intermediate representations and the presence of dropout modules in the encoder-decoder layers. The architectural analysis conducted in this research identified that while traditional Monte Carlo Dropout can be applied by activating the 37 dropout modules during inference, the absence of dropout in the detection heads means that uncertainty is captured at the feature level rather than the prediction level. This realization motivated the exploration of decoder variance as an alternative uncertainty quantification method that exploits the multi-layer transformer structure.

The second research question, regarding calibration of confidence scores, is addressed through temperature scaling, a post-hoc calibration method that operates independently of the underlying architecture. The deterministic nature of the detection heads makes temperature scaling particularly appropriate, as it recalibrates the confidence scores without requiring architectural modifications or retraining. The transformer architecture's tendency toward overconfident predictions, likely due to the sharpening effect of repeated attention operations, makes calibration especially important for this model class.

The third research question, concerning the integration of uncertainty and calibration, benefits from the modular nature of the transformer architecture. Different uncertainty quantification methods can be applied to different architectural components, such as Monte Carlo Dropout to the encoder-decoder and temperature scaling to the detection heads, and then combined through decision fusion mechanisms. The architectural separation between feature extraction, cross-modal fusion, and prediction enables these orthogonal approaches to uncertainty quantification to be composed effectively.

The final research question, regarding selective prediction for safety-critical applications, is supported by the architecture's provision of multiple uncertainty signals. The model's confidence scores, decoder layer variance, and Monte Carlo Dropout variance each reflect different aspects of prediction uncertainty that can be combined for robust risk assessment. The transformer architecture's interpretable attention mechanisms also enable post-hoc analysis of why certain predictions are uncertain, supporting the development of human-interpretable selective prediction rules.

## Summary

The GroundingDINO architecture provides a sophisticated foundation for investigating epistemic uncertainty in open-vocabulary object detection for autonomous driving applications. Its transformer-based design, combining Swin Transformer visual encoding, BERT language encoding, and cross-modal fusion through attention mechanisms, enables both high detection performance and rich uncertainty signals.

The architectural analysis conducted in this research revealed critical insights for uncertainty quantification: the placement of 37 dropout modules exclusively in transformer layers rather than detection heads, the availability of six-layer decoder representations for variance-based uncertainty estimation, and the multimodal nature of uncertainty in vision-language grounding. These findings informed the development of tailored uncertainty quantification methods that leverage the architecture's strengths while accounting for its constraints.

Understanding the network architecture deeply is essential for interpreting the uncertainty estimates produced by different methods and for appreciating why certain approaches are more suitable than others for transformer-based open-vocabulary detectors. This architectural foundation supports the subsequent experimental investigations into calibration, selective prediction, and safety-critical decision-making for autonomous driving perception systems.
