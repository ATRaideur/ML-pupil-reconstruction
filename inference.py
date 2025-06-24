import torch
import cv2
import numpy as np
from unet import UNet  # your UNet class
import matplotlib.pyplot as plt

# Load model
model = UNet()
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
model.eval()

# Load noisy image

for x in range(10):
    img = cv2.imread(f'noisy_circles/clean_{x}.png', cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        output_mask = output.squeeze().numpy()

    # Optional: Load original clean mask to compare (if available)
    mask = cv2.imread(f'noisy_circles/noisy_{x}.png', cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0 if mask is not None else None

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Noisy Input")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    if mask is not None:
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Reconstruction")
    plt.imshow(output_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
