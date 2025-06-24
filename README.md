# Circle Segmentation and Tracking with U-Net & OpenCV

This project combines real-time video processing and deep learning to detect and reconstruct circular shapes from noisy input, using a trained U-Net model. It visualizes both the detected circle and a smoothed tracking point in a live video stream.

This porject cointains 4 files : 

- dataset : generate the dataset to train the ML model (and some sample in a folder).
- unet : is the U-Net model definition.
- train_model : trains the unet model with the dataset and generate a pth file.
- inference : to test the reconstruction of a noisy circle.

##  Overview

- **Input:** A noisy video of a dark circular shape (e.g. pupil).
- **Output:** 
  - A cleaned binary mask using classical image processing.
  - A U-Net based reconstruction of the circle.
## Features
- U-Net model for semantic segmentation of noisy images.
- Morphological operations to improve classical mask generation.

  Results : 
![Figure_1](https://github.com/user-attachments/assets/cdcfac6b-ad10-4908-a002-24a99f013e5d)
![Figure_2](https://github.com/user-attachments/assets/c15881a0-dd60-48d9-acb4-e968bc2914e1)
![Figure_3](https://github.com/user-attachments/assets/b81acb9f-571c-4baa-8839-5f01ec632beb)
![Figure_4](https://github.com/user-attachments/assets/e7b53702-afc2-499d-9c3a-dfa409692d97)


the main porblem with this technique of reconstruction of the pupil is that it takes a long time (~1s) to procces each 
frames, it is unusable on a real time system because of the procces time of the frames.
