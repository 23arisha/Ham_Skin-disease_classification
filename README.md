# Skin Disease Classification Project

## Overview
The **Skin Disease Classification Project** is a deep learning initiative aimed at diagnosing skin conditions using medical imaging. The project leverages the HAM10000 dataset, a publicly available collection of skin lesion images, to classify seven types of skin conditions with the aid of a custom Convolutional Neural Network (CNN) model.

### Key Features:
- **Dataset**: HAM10000, containing thousands of dermoscopic images labeled with conditions.
- **Target Classes**:
  - **BKL**: Benign keratosis-like lesions
  - **NV**: Melanocytic nevi (moles)
  - **DF**: Dermatofibroma
  - **MEL**: Melanoma (a type of skin cancer)
  - **VASC**: Vascular lesions
  - **BCC**: Basal cell carcinoma
  - **AKIEC**: Actinic keratoses and intraepithelial carcinoma
- **Deep Learning Model**: A custom CNN designed for multi-class classification.
- **Tools and Libraries**: PyTorch, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

---

## Objectives
- **Automate skin disease detection** to aid dermatologists in diagnosing conditions with greater accuracy and efficiency.
- **Improve diagnostic accuracy** for life-threatening conditions such as melanoma.
- **Facilitate balanced data sampling** to address class imbalances present in medical datasets.

---

## Workflow
1. **Data Preparation**:
   - Load metadata and image paths from the HAM10000 dataset.
   - Label encode the skin condition classes.
   - Balance the dataset by resampling underrepresented classes.

2. **Custom Dataset and DataLoader**:
   - Implement a PyTorch dataset class for loading and preprocessing images.
   - Normalize images and resize them to a uniform size of 128x128 pixels.

3. **Model Architecture**:
   - Build a CNN with three convolutional layers, dropout regularization, and two fully connected layers for classification into seven classes.

4. **Training**:
   - Train the model using the cross-entropy loss function and Adam optimizer.
   - Evaluate performance after each epoch by calculating training loss and accuracy.

5. **Testing and Evaluation**:
   - Evaluate the model on a test set to calculate accuracy.
   - Generate a confusion matrix to analyze classification performance across all classes.

---

## Results
- **Training Accuracy**: Achieved 96.11% on the training set after 15 epochs.
- **Testing Accuracy**: Reached 75.09% on the test set.
- **Confusion Matrix**: Visualized class-wise performance and identified areas for improvement.

---

## Future Work
- **Enhance Model Accuracy**:
  - Explore advanced architectures like ResNet or EfficientNet for better feature extraction.
  - Implement data augmentation techniques to improve generalization.
- **Expand Dataset**:
  - Collect additional diverse and high-quality dermoscopic images.
- **Integration**:
  - Develop a web-based application or API for real-time predictions in clinical settings.

---
