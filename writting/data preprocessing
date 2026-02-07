## Data Preprocessing

### Dataset Selection and Overview

This research utilizes the Berkeley DeepDrive 100K (BDD100K) dataset, a large-scale, diverse driving video dataset designed for heterogeneous multitask learning in Advanced Driver Assistance Systems (ADAS). The BDD100K dataset was specifically selected due to its comprehensive representation of real-world autonomous driving scenarios, featuring diverse geographical locations, weather conditions, and times of day. The dataset contains 100,000 high-resolution images (1280×720 pixels) with rich annotations for object detection tasks, making it particularly suitable for evaluating uncertainty quantification methods in safety-critical applications.

For this study, we focus exclusively on the validation split of BDD100K, which comprises 10,000 annotated images. The validation split provides a robust testbed with 10 object categories relevant to autonomous driving: person, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, and traffic sign. This category set encompasses both large objects (vehicles) and small objects (traffic signs and lights), enabling comprehensive evaluation across different object scales and detection difficulties.

### Format Conversion and Standardization

The BDD100K dataset originally employs a custom annotation format that differs from the widely-adopted COCO (Common Objects in Context) format used by most modern object detection frameworks. To ensure compatibility with standard evaluation tools and facilitate reproducible benchmarking, a comprehensive format conversion pipeline was implemented.

The conversion process begins by parsing the BDD100K JSON annotation files, which store bounding box coordinates in absolute pixel values (x1, y1, x2, y2) format. These annotations are systematically transformed into COCO format, which requires bounding boxes in (x, y, width, height) representation, where (x, y) denotes the top-left corner coordinates. Each annotation is assigned a unique identifier, mapped to the corresponding image ID, and associated with the appropriate category ID based on a predefined category mapping.

A critical aspect of this conversion involves handling categorical variations in the original BDD100K labels. The dataset uses alternative naming conventions for certain categories, requiring alias resolution. For instance, the category labeled as "bike" in BDD100K is standardized to "bicycle" to maintain consistency with common object detection taxonomies. Additionally, the "pedestrian" category in BDD100K is mapped to "person" to align with standard detection benchmarks.

The conversion pipeline validates each bounding box to ensure geometric validity, filtering out annotations with non-positive dimensions that could arise from annotation errors or edge cases. This validation step is essential for maintaining data quality and preventing computational errors during subsequent evaluation phases. The resulting COCO-formatted annotations include comprehensive metadata such as image dimensions, bounding box areas, and crowd flags, ensuring full compatibility with standard evaluation protocols.

### Dataset Splitting Strategy

A strategic data splitting approach was implemented to support both calibration and unbiased evaluation of uncertainty quantification methods. The 10,000-image validation split was partitioned into two distinct subsets using stratified random sampling with a fixed random seed (seed=42) to ensure reproducibility across experiments.

The primary split consists of a calibration set (val_calib) containing 8,000 images (80% of the validation data) and an evaluation set (val_eval) comprising 2,000 images (20% of the validation data). This 80-20 split ratio was deliberately chosen to provide sufficient data for robust calibration of temperature scaling methods while reserving an adequate holdout set for final performance assessment. The calibration set is exclusively used for optimizing temperature parameters in temperature scaling methods, while the evaluation set remains completely unseen during any training or calibration procedures, serving as a true test set for reporting final metrics.

To ensure statistical validity, the splitting procedure employs random shuffling of image identifiers before partitioning, preventing any systematic bias that might arise from the original ordering of images in the dataset. Importantly, the split is performed at the image level, not at the annotation level, ensuring that all annotations belonging to the same image remain together in the same subset. This image-level splitting is crucial for maintaining the natural distribution of object co-occurrences and scene compositions present in real-world driving scenarios.

The split generation process includes comprehensive verification steps to confirm zero overlap between calibration and evaluation sets. Statistical analyses confirm that both subsets maintain representative distributions of object categories, object sizes, and scene complexities, ensuring that evaluation results generalize across diverse driving conditions.

### Text Prompt Engineering for Open-Vocabulary Detection

Unlike traditional closed-set object detectors that operate on fixed category lists, the GroundingDINO model employed in this research requires natural language text prompts to guide the detection process. The design of these text prompts significantly influences detection performance, as the model relies on cross-modal alignment between visual features and linguistic representations.

A prompt file (bdd100k.txt) was created containing simple, canonical class names for the 10 BDD100K categories. Each category name is specified on a separate line using lowercase, singular form without articles or additional descriptors. This minimalist prompt design was deliberately chosen to avoid introducing linguistic ambiguities or biases that could affect detection consistency. For example, the prompt uses "person" rather than variations like "pedestrian," "human," or "individual," ensuring semantic clarity and reducing potential confusion in the vision-language alignment process.

The prompt engineering strategy prioritizes semantic precision over linguistic complexity. Categories are presented in their most common lexical form, matching the terminology used in the COCO dataset and other standard benchmarks. This alignment facilitates fair comparison with baseline results and ensures that detection performance reflects the model's visual understanding rather than prompt engineering artifacts.

During inference, the complete prompt string is constructed by concatenating all category names with period separators, forming a single text query: "person . rider . car . truck . bus . train . motorcycle . bicycle . traffic light . traffic sign ." This format adheres to the expected input specification of GroundingDINO, where the period delimiter enables the model to segment the prompt into individual category queries while maintaining the full vocabulary context.

### Image Preprocessing and Normalization

The GroundingDINO model architecture imposes specific preprocessing requirements to ensure proper functioning of its vision-language components. All input images undergo standardized preprocessing steps that align with the model's training configuration.

First, images are loaded from disk and decoded into RGB format with pixel values in the range [0, 255]. The native resolution of BDD100K images (1280×720 pixels) is preserved during loading to maintain the full spatial detail present in the original captures. No resizing or cropping operations are applied to the input images, as GroundingDINO's Swin Transformer backbone is designed to handle variable input resolutions through adaptive pooling mechanisms.

Pixel intensity normalization is performed using ImageNet statistics, which is standard practice for models pretrained on ImageNet or its derivatives. Each color channel is independently normalized by subtracting the channel-wise mean and dividing by the channel-wise standard deviation. Specifically, RGB channels are normalized using means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225]. This normalization ensures that input features have zero mean and unit variance, facilitating stable gradient flow during inference and aligning with the statistics of the pretraining data.

The normalized image tensors are then formatted according to PyTorch conventions, with dimensions arranged as [batch_size, channels, height, width]. All preprocessing operations maintain float32 precision to preserve numerical accuracy throughout the detection pipeline. No data augmentation techniques are applied during inference, as this study focuses on uncertainty quantification under standard test conditions rather than robustness to augmentations.

### Ground Truth Annotation Processing

The COCO-formatted ground truth annotations undergo additional processing to facilitate efficient evaluation and error analysis. Each annotation record contains essential information including bounding box coordinates, category identifiers, object areas, and image associations.

Bounding boxes in the ground truth are stored in COCO format (x, y, width, height) and are converted to normalized coordinates for certain evaluation metrics. The area field, computed as the product of box width and height, serves as a key attribute for categorizing objects by size. Following COCO evaluation protocols, objects are classified into three size categories: small objects (area < 32² pixels), medium objects (32² ≤ area < 96² pixels), and large objects (area ≥ 96² pixels). This size-based categorization enables detailed performance analysis across different object scales, which is particularly important for autonomous driving applications where small objects like distant traffic signs pose significant detection challenges.

The crowd flag in annotations is utilized to identify groups of objects that are difficult to segment individually, such as crowds of people or dense traffic scenarios. These annotations are handled appropriately during evaluation, with instance-level precision metrics adjusted accordingly.

### Data Quality Validation and Statistics

Comprehensive data quality checks were performed to ensure the integrity of the preprocessed dataset. Validation procedures include verification of image file existence, confirmation of readable image formats, validation of bounding box coordinates to ensure they fall within image boundaries, and checking for degenerate boxes with zero or negative dimensions.

Statistical analysis of the preprocessed dataset reveals important characteristics that inform subsequent experimental design. The distribution of object categories is highly imbalanced, reflecting real-world driving scenarios where cars constitute the vast majority of detections (approximately 60% of all annotations), followed by traffic signs (23%), traffic lights (11%), and pedestrians (3%). This class imbalance necessitates careful consideration during evaluation, motivating the use of per-category metrics in addition to mean average precision.

Object size distribution analysis shows that small objects (primarily traffic signs and distant traffic lights) represent a significant proportion of annotations, highlighting the importance of multi-scale detection capabilities. The average number of annotated objects per image is approximately 11, with substantial variance across scenes ranging from sparse highway scenarios to dense urban intersections with dozens of objects.

### Data Loading and Batching Strategy

Due to the variable resolution support of GroundingDINO and the diversity of object counts across images, a single-image batch processing strategy is employed during inference. Each image is processed independently with batch_size=1, avoiding the need for padding or resizing operations that could introduce artifacts or alter object scales.

For computational efficiency during multi-pass inference required by MC-Dropout uncertainty estimation, images are loaded on-demand from disk rather than preloaded into memory. This streaming approach manages GPU memory effectively while allowing the processing of the complete evaluation set without memory constraints. Image paths and corresponding annotations are organized in indexed data structures that enable efficient random access during evaluation.

The data loading pipeline incorporates error handling mechanisms to gracefully skip corrupted images or malformed annotations that might have escaped initial validation checks. Detailed logging tracks which images are successfully processed and which encounter errors, ensuring complete traceability of the evaluation process.

### Preprocessing Pipeline Integration

The complete preprocessing pipeline is implemented as a modular, reproducible workflow that can be executed independently of subsequent experimental phases. All preprocessing scripts and notebooks document exact versions of libraries used, random seeds for splitting operations, and configuration parameters for format conversion. This documentation ensures full reproducibility of the preprocessed data and facilitates future extensions of the experimental framework.

The preprocessed data artifacts, including COCO JSON files and validation split definitions, are stored in a standardized directory structure that maintains clear separation between raw source data, intermediate conversion outputs, and final processed datasets ready for model inference. This organization enables efficient data management and reduces the risk of using incorrect data splits during different phases of experimentation.

The preprocessing phase establishes a solid foundation for all subsequent uncertainty quantification and calibration experiments, ensuring that observed performance differences across methods reflect genuine algorithmic properties rather than data handling artifacts or preprocessing inconsistencies.
