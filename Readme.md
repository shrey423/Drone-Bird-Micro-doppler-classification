# Drone-Bird Micro-Doppler Classification

## Project Overview
This project aims to solve the challenge of distinguishing between drones and birds using their micro-doppler signatures in radar systems. The project originated from the Sih Hackathon 2025 problem statement, which highlighted the difficulty in differentiating between unmanned aerial vehicles (UAVs) and birds based on their radar signatures.

## Technical Approach
### Research Analysis and Model Design
Before developing the model, extensive research was conducted on existing approaches for micro-doppler signature classification. Key findings and challenges identified:

1. **Challenges in Traditional Approaches**:
   - Most existing models struggled with class imbalance
   - Limited generalization to unseen data
   - High computational requirements
   - Difficulty in capturing subtle micro-doppler patterns

2. **Technical Decisions**:
   - Chose a lightweight CNN architecture to handle real-time applications
   - Implemented attention mechanisms (SE blocks) for better feature extraction
   - Used mixup data augmentation to improve generalization
   - Developed a custom loss function (Focal Loss with Label Smoothing) for better convergence

## Dataset
The project utilizes the DIAT-MSAT Micro-Doppler Signature Dataset for Small Unmanned Aerial Vehicle (SUAV) from IEEE DataPort. This dataset contains micro-doppler signatures of various UAVs and birds, enabling the development of robust classification models.

## Model Architecture
The model architecture is designed to address specific challenges:

1. **Attention Mechanisms**:
   - Implemented SE blocks with improved reduction ratio (8 instead of 16)
   - Added feature recalibration at multiple layers
   - Enhanced feature extraction for micro-doppler patterns

2. **Data Augmentation**:
   - Mixup augmentation with balanced alpha (0.4)
   - Aggressive data augmentation including:
     - Rotation up to 30 degrees
     - Width/Height shifts of 30%
     - Shear range of 0.2
     - Zoom range of 0.3
     - Horizontal and vertical flips

3. **Loss Function**:
   - Custom Focal Loss with Label Smoothing
   - Gamma=2.0 and Alpha=0.25 for better convergence
   - Label smoothing of 0.1 to prevent overconfidence

4. **Architecture Improvements**:
   - Residual connections for deeper networks
   - Separable Conv2D for reduced computational cost
   - Two-stage classification head for better feature learning
   - Dropout layers for regularization

## Challenges and Solutions

1. **Class Imbalance**:
   - Implemented class weights using sklearn's compute_class_weight
   - Used balanced sampling during training
   - Added label smoothing to prevent overfitting

2. **Overfitting Prevention**:
   - Added multiple dropout layers (0.4-0.5)
   - Implemented early stopping with patience
   - Used ReduceLROnPlateau for adaptive learning rate

3. **Computational Efficiency**:
   - Used SeparableConv2D instead of regular Conv2D
   - Implemented lightweight architecture with residual connections
   - Optimized GPU memory usage

## Project Structure
- `finalmodel.py`: Main implementation file containing the machine learning model
- `dataset_analysis.py`: Code for analyzing and preprocessing the dataset
- `check.py`: Testing and validation scripts
- `test.py`: Additional testing utilities
- `class_distribution.png`: Visualization of class distribution
- `class_distribution_boxplot.png`: Box plot of class distributions
- `class_proportions.png`: Proportion visualization of different classes
- `model_outputs/`: Directory containing model predictions and outputs
- `results_folder/`: Directory for storing experimental results
- `final_result/`: Directory for final project results

## Features
- Micro-doppler signature analysis with attention mechanisms
- Machine learning-based classification with custom loss functions
- Comprehensive dataset analysis and visualization
- Advanced data augmentation techniques
- Performance metrics evaluation with class imbalance handling
- Interactive visualization of prediction results
- Binary classification with confidence scores
- Detailed class analysis for micro-doppler patterns

## Visualization Capabilities
The project includes a powerful visualization system that can:

1. **Micro-Doppler Signature Analysis**:
   - Analyze and classify micro-doppler signatures from radar images
   - Support for both single image and batch processing
   - Temperature scaling for improved probability distribution
   - Confidence aggregation for binary classification

2. **Prediction Visualization**:
   - Side-by-side image display with prediction results
   - Top N prediction probabilities visualization
   - Confidence scores for both bird and drone classification
   - Detailed class analysis showing specific drone types

3. **Classification Groups**:
   - Drone Types:
     - 2-blade rotor drones
     - 3-blade rotor drones
     - Various drone models
   - Bird Classification:
     - Pure bird signatures
     - Ambiguous cases (bird+drone combinations)

4. **GUI Interface**:
   - Interactive GUI for single image classification
   - Batch processing capabilities
   - Real-time progress updates
   - Export of results in multiple formats

## Getting Started
1. Clone the repository
2. Install required dependencies (see requirements.txt)
3. Download the DIAT-MSAT dataset from IEEE DataPort
4. Run `dataset_analysis.py` to preprocess the data
5. Train the model using `finalmodel.py`
6. Use `check.py` for visualization and prediction:
   ```bash
   python check.py --model_path your_model.h5 --image_path test_image.jpg
   ```

## Using the Visualization System

1. **Single Image Analysis**:
   ```bash
   python check.py --model_path your_model.h5 --image_path path/to/image.jpg
   ```
   - Generates visualizations of prediction probabilities
   - Shows both detailed and binary classification results
   - Saves results to output directory

2. **Batch Processing**:
   ```bash
   python check.py --model_path your_model.h5 --image_dir path/to/images --output_dir results
   ```
   - Processes multiple images in a directory
   - Generates summary CSV file
   - Creates visualization for each image
   - Saves detailed results for each prediction

3. **Interactive GUI**:
   ```bash
   python check.py --gui
   ```
   - Opens an interactive GUI interface
   - Drag and drop images for analysis
   - Real-time visualization updates
   - Batch processing capabilities

## Prediction Results Format
The system provides three levels of classification:

1. **Detailed Classification**:
   - Specific drone types (e.g., 2_blade_rotor_1, 3_long_blades_rotor)
   - Bird types
   - Ambiguous cases (bird+drone combinations)

2. **Binary Classification**:
   - Bird vs Drone
   - Confidence scores for each class
   - Aggregated from detailed predictions

3. **Confidence Scores**:
   - Temperature scaling for probability distribution
   - Aggregated confidence for binary classification
   - Top N predictions with confidence percentages

## Output Files
The visualization system generates:
- Prediction visualizations (PNG format)
- Detailed results text files
- Batch prediction summary CSV
- Confidence score reports
- Micro-doppler pattern analysis results

## Results
The project includes various visualizations and analysis results:
- Class distribution analysis
- Performance metrics
- Model validation results
- Attention mechanism visualizations
- Data augmentation examples

## gui 
![Alt text](https://github.com/shrey423/Drone-Bird-Micro-doppler-classification/blob/main/images/Screenshot%202025-04-02%20224834.png?raw=true)
![Alt text](https://github.com/shrey423/Drone-Bird-Micro-doppler-classification/blob/main/images/Screenshot%202025-04-02%20224923.png?raw=true)

## References
- DIAT-MSAT Micro-Doppler Signature Dataset: https://ieee-dataport.org/documents/diat-msat-micro-doppler-signature-dataset-small-unmanned-aerial-vehicle-suav
- Sih Hackathon 2025 Problem Statement
- Research papers on micro-doppler signature analysis
- Implementations of attention mechanisms in CNNs

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- IEEE DataPort for providing the DIAT-MSAT dataset
- Sih Hackathon 2025 for inspiring this project
- All contributors and team members involved in the development
- Researchers whose work was referenced during model development

## Future Work
1. Implement real-time processing capabilities
2. Add support for additional radar signature types
3. Improve model efficiency for edge devices
4. Add more sophisticated attention mechanisms
5. Implement ensemble learning approaches

## Contact
For any questions or collaboration opportunities, please contact shreyraj12210@gmail.com
