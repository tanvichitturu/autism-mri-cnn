Autism Detection using Brain MRI and CNN
Overview

This project presents a deep learning–based approach for early screening of Autism Spectrum Disorder (ASD) using structural brain MRI scans. Raw medical imaging data in NIfTI (.nii) format is preprocessed into 2D slices, and a Convolutional Neural Network (CNN) is trained to classify autistic and non-autistic subjects.

The system is designed as a decision-support tool, not a clinical diagnostic replacement.

Pipeline

Load 3D MRI brain scans (.nii files)

Extract informative middle brain slices

Normalize and resize images (128×128)

Construct labeled dataset

Train CNN classifier

Aggregate slice predictions to produce patient-level classification

Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

NiBabel (medical imaging)

Scikit-learn

Model

A CNN architecture with three convolutional layers and max-pooling layers is used to learn structural brain patterns associated with ASD. Final predictions are obtained by averaging probabilities across multiple slices of the same subject.

Results

Training Accuracy: ~99%

Validation Accuracy: ~75%

The difference indicates overfitting due to slice-level correlation between images from the same subject, which is a common challenge in medical imaging tasks.

Important Note

This project does not diagnose autism.
It demonstrates how machine learning can assist clinicians by identifying structural patterns correlated with ASD. Clinical diagnosis must still rely on professional behavioral and neurological assessment.

How to Run
# create virtual environment
python -m venv mri_env
.\mri_env\Scripts\activate

# install dependencies
pip install -r requirements.txt

# create dataset from MRI
python create_dataset.py

# train model
python train_cnn.py

# predict new scan
python predict.py
Future Improvements

Patient-level train/test split

Data augmentation

3D CNN architecture

Cross-site validation using ABIDE dataset
