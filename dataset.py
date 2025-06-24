import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from scipy.interpolate import splprep, splev

class NoisyCircleDataset(Dataset):
    def __init__(self, size=512, n_samples=100, super_res_factor=4, noise_intensity=0.5):

        self.size = size
        self.n_samples = n_samples
        self.super_res = size * super_res_factor  # Render at higher resolution
        self.min_radius = int(size * 0.15)
        self.max_radius = int(size * 0.4)
        self.noise_intensity = noise_intensity

    def __len__(self):
        return self.n_samples

    def generate_smooth_shape(self, center, avg_radius, irregularity=0.6, spikiness=0.2, n_points=16):
        n_points = max(n_points, 12)
        points = []
        angle = 0
        for _ in range(n_points):
            r_i = avg_radius * (1 + random.uniform(-spikiness, spikiness))
            r_i = np.clip(r_i, avg_radius*0.7, avg_radius*1.3)
            x = center[0] + r_i * np.cos(angle)
            y = center[1] + r_i * np.sin(angle)
            points.append([x, y])
            angle += 2*np.pi/n_points * (1 + random.uniform(-irregularity, irregularity))

        points = np.array(points)

        try:
            tck, u = splprep(points.T, s=0, per=1)
            u_new = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_new, tck)
            smooth_points = np.vstack([x_new, y_new]).T
            return smooth_points
        except:
            return points

    def draw_anti_aliased(self, canvas, shape_points, value):
        hr_canvas = np.zeros((self.super_res, self.super_res), dtype=np.uint8)
        hr_scale = self.super_res / self.size
        hr_points = (shape_points * hr_scale).astype(np.int32)
        cv2.fillPoly(hr_canvas, [hr_points], 255)
        downsampled = cv2.resize(hr_canvas, (self.size, self.size),
                                 interpolation=cv2.INTER_AREA)
        canvas[downsampled > 0] = value

    def add_random_white_blobs(self, img, max_blobs=10, max_radius_ratio=0.2):
        h, w = img.shape
        num_blobs = random.randint(0, int(max_blobs * self.noise_intensity))

        for _ in range(num_blobs):
            # Random position anywhere in the image
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)

            # Random size (relative to image dimensions)
            max_r = int(min(h, w) * max_radius_ratio * random.uniform(0.5, 1.5))
            blob_r = random.randint(int(max_r * 0.1), max_r)

            # Random irregularity
            irregularity = random.uniform(0.4, 0.9)
            spikiness = random.uniform(0.3, 0.7)

            # Generate and draw the blob
            pts = self.generate_smooth_shape((cx, cy), blob_r, irregularity, spikiness)
            self.draw_anti_aliased(img, pts, 1.0)  # White blob (value=1)

        return img

    def __getitem__(self, idx):
        img = np.zeros((self.size, self.size), dtype=np.float32)
        mask = np.zeros_like(img)

        # Generate main ellipse
        center = (
            random.randint(self.max_radius, self.size - self.max_radius),
            random.randint(self.max_radius, self.size - self.max_radius)
        )
        axis1 = random.randint(self.min_radius, self.max_radius)
        axis2 = random.randint(int(axis1*0.7), axis1)
        angle = random.uniform(0, 360)

        # Draw anti-aliased ellipse
        ellipse_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.ellipse(ellipse_mask, center, (axis1, axis2), angle, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        mask = ellipse_mask.astype(np.float32) / 255.0
        img = mask.copy()

        # Add white blobs (randomly anywhere in the image)
        img = self.add_random_white_blobs(img, max_blobs=15, max_radius_ratio=0.15)

        return torch.tensor(img).unsqueeze(0), torch.tensor(mask).unsqueeze(0)

# Generate samples
SAVE_DIR = "noisy_circles"
os.makedirs(SAVE_DIR, exist_ok=True)
dataset = NoisyCircleDataset(size=512, n_samples=100, noise_intensity=0.7)

for i in range(10):
    noisy, clean = dataset[i]
    cv2.imwrite(f"{SAVE_DIR}/noisy_{i}.png", (noisy.squeeze().numpy() * 255).astype(np.uint8))
    cv2.imwrite(f"{SAVE_DIR}/clean_{i}.png", (clean.squeeze().numpy() * 255).astype(np.uint8))