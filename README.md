Monkeypox Detection using MobileNet & TensorFlow
Overview
This project implements a Monkeypox Detection model using MobileNet as a feature extractor and a custom classification head. The model is trained on a dataset of Monkeypox images and uses transfer learning to classify infected vs. non-infected cases.

Project Structure
Dataset: Images are organized into Train, Validation, and Test directories.
Preprocessing: Image augmentation and normalization using ImageDataGenerator.
Model: MobileNet-based CNN with additional dense layers.
Training: Optimized with Adam optimizer and monitored using a learning rate scheduler.
Evaluation: Metrics like accuracy, precision, recall, F1-score, and confusion matrix are computed.
Requirements
Install the required dependencies before running the code:

bash
Copy
Edit
pip install tensorflow numpy scikit-learn
Dataset Structure
Ensure your dataset follows this directory structure:

Copy
Edit
mpox/
│── Fold1/
│   ├── Train/
│   │   ├── Infected/
│   │   ├── Non-Infected/
│   ├── Val/
│   │   ├── Infected/
│   │   ├── Non-Infected/
│   ├── Test/
│   │   ├── Infected/
│   │   ├── Non-Infected/
Each folder should contain images of Monkeypox-infected and non-infected cases.

How to Run
1. Train the Model
Run the script to start training:

bash
Copy
Edit
python train.py
The model will be trained for 4 epochs with binary cross-entropy loss.
Learning rate reduces automatically if validation loss stagnates.
2. Evaluate the Model
After training, the model is evaluated on the test set, computing:

Accuracy
Precision, Recall, and F1 Score
Confusion Matrix
Class-wise Accuracy
Key Features
✅ Uses MobileNet as a base model for transfer learning
✅ Image Augmentation (rotation, shift, shear, zoom, flip)
✅ Binary Classification (Monkeypox vs. Non-infected)
✅ Optimized with Adam Optimizer & Learning Rate Scheduling
✅ Comprehensive Evaluation Metrics

Results
Overall Accuracy: XX% (after training)
Precision: XX%
Recall: XX%
F1 Score: XX%

Future Improvements
🔹 Train for more epochs for better performance
🔹 Implement Grad-CAM for explainability
🔹 Hyperparameter tuning
