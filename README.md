# STN_Final_Term
This project integrates a Spatial Transformer Network (STN) into a YOLOv10 architecture to improve object detection performance on chest X-ray films. The goal is to accurately detect abnormal pathological signs (such as nodules, consolidations, and other anomalies) on chest X-rays
Key Features
STN Integration:
Incorporates a Spatial Transformer Network to align input images. This spatial normalization helps the model handle variations in position, rotation, and scale, making it more robust to differences in patient positioning and image quality.

YOLOv10 Backbone:
Leverages the YOLOv10 detection framework, which is known for its real-time performance and accuracy, and adapts it to work with medical images.

Detection of Chest Abnormalities:
Custom-tailored to detect abnormal pathological signs on chest X-ray films, enabling the automated detection of potentially life-threatening conditions.

End-to-End Training:
The integrated system is trained end-to-end, ensuring that the STN and YOLO components are jointly optimized for the task.

Project Structure
Models: Contains the modified YOLOv10 model with STN integration.

Datasets: Sample datasets and annotation files for chest X-ray images.

Scripts: Training, evaluation, and inference scripts.

Notebooks: Jupyter notebooks for data visualization and model performance analysis.

Utilities: Helper functions for data preprocessing, bounding box transformation, and visualization.

Requirements
Python 3.8+

PyTorch 1.9+

TorchVision

OpenCV

NumPy

Matplotlib

Other dependencies as listed in requirements.txt
