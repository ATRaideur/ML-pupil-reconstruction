import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from scipy.interpolate import splprep, splev

class NoisyCircleDataset(Dataset):
    def __init__(self, size=512, n_samples=100, super_res_factor=4, noise_intensity=0.5):
        """
        Args:
            size: Output image size
            n_samples: Number of samples
            super_res_factor: Render at higher resolution then downsample (anti-aliasing)
            noise_intensity: Control the amount of noise (0 to 1)
        """
        self.size = size
        self.n_samples = n_samples
        self.super_res = size * super_res_factor  # Render at higher resolution
        self.min_radius = int(size * 0.15)
        self.max_radius = int(size * 0.4)
        self.noise_intensity = noise_intensity

    def __len__(self):
        return self.n_samples

    def generate_smooth_shape(self, center, avg_radius, irregularity=0.6, spikiness=0.2, n_points=16):
        """Generate smooth organic shapes with anti-aliased edges"""
        # Generate more points for smoother shapes
        n_points = max(n_points, 12)

        # Create base shape
        points = []
        angle = 0
        for _ in range(n_points):
            # Smoother radius variation
            r_i = avg_radius * (1 + random.uniform(-spikiness, spikiness))
            r_i = np.clip(r_i, avg_radius*0.7, avg_radius*1.3)
            x = center[0] + r_i * np.cos(angle)
            y = center[1] + r_i * np.sin(angle)
            points.append([x, y])
            angle += 2*np.pi/n_points * (1 + random.uniform(-irregularity, irregularity))

        points = np.array(points)

        # High-quality spline interpolation
        try:
            tck, u = splprep(points.T, s=0, per=1)
            u_new = np.linspace(0, 1, 500)  # More points for smoother curves
            x_new, y_new = splev(u_new, tck)
            smooth_points = np.vstack([x_new, y_new]).T
            return smooth_points
        except:
            return points

    def draw_anti_aliased(self, canvas, shape_points, value):
        """Draw anti-aliased shapes using super-resolution"""
        # Create high-res canvas
        hr_canvas = np.zeros((self.super_res, self.super_res), dtype=np.uint8)
        hr_scale = self.super_res / self.size

        # Scale points to high-res
        hr_points = (shape_points * hr_scale).astype(np.int32)
        cv2.fillPoly(hr_canvas, [hr_points], 255)

        # Downsample with anti-aliasing
        downsampled = cv2.resize(hr_canvas, (self.size, self.size),
                                 interpolation=cv2.INTER_AREA)
        canvas[downsampled > 0] = value

    def add_speckle_noise(self, img, intensity):
        """Add speckle noise to the image"""
        noise = np.random.randn(*img.shape) * intensity
        noisy_img = img * (1 + noise)
        return np.clip(noisy_img, 0, 1)

    def add_random_black_spots(self, img, mask, max_spots=20, max_spot_size=0.1):
        """Add random black spots to the image within the mask area"""
        h, w = img.shape
        num_spots = random.randint(1, max_spots)
        
        for _ in range(num_spots):
            # Find a random position within the mask
            ys, xs = np.where(mask > 0.5)
            if len(ys) == 0:
                continue
                
            idx = random.randint(0, len(ys)-1)
            center_x, center_y = xs[idx], ys[idx]
            
            # Random spot size
            spot_size = random.randint(1, int(max_spot_size * min(h, w)))
            
            # Random shape (circle or irregular blob)
            if random.random() > 0.5:
                # Circle
                cv2.circle(img, (center_x, center_y), spot_size, 0, -1)
            else:
                # Irregular blob
                pts = self.generate_smooth_shape((center_x, center_y), spot_size)
                self.draw_anti_aliased(img, pts, 0)
        
        return img

    def add_texture_noise(self, img, mask):
        """Add texture-like noise to the circle"""
        # Create Perlin noise or similar texture
        noise = np.random.rand(*img.shape) * 0.3 * self.noise_intensity
        
        # Apply only within the mask area
        img[mask > 0] = np.clip(img[mask > 0] - noise[mask > 0], 0, 1)
        
        return img

    def add_edge_erosion(self, img, mask, iterations=3):
        """Erode the edges of the circle to make it look damaged"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        eroded = cv2.erode(mask, kernel, iterations=iterations)
        edge_damage = mask - eroded
        
        # Add random damage to edges
        damage_mask = (edge_damage > 0) & (np.random.rand(*img.shape) < 0.3 * self.noise_intensity)
        img[damage_mask] = 0
        
        return img

    def __getitem__(self, idx):
        # Initialize high-quality images
        img = np.zeros((self.size, self.size), dtype=np.float32)
        mask = np.zeros_like(img)

        # Generate main ellipse
        center = (
            random.randint(self.max_radius, self.size - self.max_radius),
            random.randint(self.max_radius, self.size - self.max_radius)
        )
        axis1 = random.randint(self.min_radius, self.max_radius)
        axis2 = random.randint(int(axis1*0.7), axis1)  # Less eccentric
        angle = random.uniform(0, 360)

        # Draw anti-aliased ellipse
        ellipse_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.ellipse(ellipse_mask, center, (axis1, axis2), angle, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        mask = ellipse_mask.astype(np.float32) / 255.0
        img = mask.copy()

        # Add smooth defects (holes)
        n_defects = random.randint(3, 8)
        for _ in range(n_defects):
            avg_r = random.randint(int(self.min_radius*0.15), int(self.min_radius*0.3))
            cx = random.randint(center[0] - axis1, center[0] + axis1)
            cy = random.randint(center[1] - axis1, center[1] + axis1)

            if (0 < cx < self.size and 0 < cy < self.size):
                pts = self.generate_smooth_shape((cx, cy), avg_r)
                self.draw_anti_aliased(img, pts, 0)

        # Add more noise and black spots
        img = self.add_random_black_spots(img, mask, 
                                        max_spots=int(20 * self.noise_intensity),
                                        max_spot_size=0.15 * self.noise_intensity)
        
        # Add texture noise
        img = self.add_texture_noise(img, mask)
        
        # Add edge erosion
        img = self.add_edge_erosion(img, mask, iterations=int(3 * self.noise_intensity))
        
        # Add speckle noise
        img = self.add_speckle_noise(img, 0.1 * self.noise_intensity)
        
        return torch.tensor(img).unsqueeze(0), torch.tensor(mask).unsqueeze(0)

# Generate samples
SAVE_DIR = "noisy_circles"
os.makedirs(SAVE_DIR, exist_ok=True)
dataset = NoisyCircleDataset(size=512, n_samples=100, noise_intensity=0.8)  # Try values 0-1

for i in range(10):
    noisy, clean = dataset[i]
    cv2.imwrite(f"{SAVE_DIR}/noisy_{i}.png", (noisy.squeeze().numpy() * 255).astype(np.uint8))
    cv2.imwrite(f"{SAVE_DIR}/clean_{i}.png", (clean.squeeze().numpy() * 255).astype(np.uint8))
