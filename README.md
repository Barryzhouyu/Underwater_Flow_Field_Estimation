## **Underwater Flow Field Estimation Using Vision-Based Motion Tomography (VMT)**

<img width="870" alt="Holo_sce" src="https://github.com/user-attachments/assets/5901c498-c05c-4db0-b2c2-86ef10728ce3">


### Overview

This repository contains the implementation of our vision-based learning method for underwater flow field estimation. The approach is designed to enhance Autonomous Underwater Vehicle (AUV) navigation by leveraging sporadically available visual sensor data. Our method allows AUVs to estimate unknown flow fields in real-time, even in the absence of direct localization information.

### Abstract

Underwater flow field estimation is critical for marine robot missions, as the stability and safety of AUVs are heavily influenced by ocean currents. We introduce a Vision-based Motion Tomography (VMT) algorithm that uses a lightweight Convolutional Neural Network (CNN) to predict the flow field from captured visual data. This method addresses the limitations of high-cost physical models and the inefficiencies of traditional navigation techniques that require continuous sensor input.

### Features

•	Vision-Based Learning: Uses sporadic visual observations from the AUV’s onboard camera for flow field estimation.
•	CNN-Based Prediction: Employs a CNN model to extract meaningful features from visual data and predict flow fields.
•	Simulation Results: Demonstrates the accuracy of our approach compared to traditional flow estimation techniques.
•	Integration with NeRF: Our model is flexible and can be integrated with Neural Radiance Fields (NeRF) for future enhancements.
