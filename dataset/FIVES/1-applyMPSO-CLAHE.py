import cv2
import numpy as np
import random
import glob
import os

# --------------------------------------------------
# 1. CLAHE Utility Functions for Color Images
# --------------------------------------------------
def apply_clahe_color(image, clip_limit, tile_grid_size):
    """
    Apply CLAHE on the luminance channel of a color image.
    The image is assumed to be in BGR format.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                            tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    image_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image_enhanced

def fitness_function_color(image, clip_limit, tile_grid_size):
    """
    Compute a fitness score for a color image based on the variance
    of the L channel after applying CLAHE.
    """
    enhanced = apply_clahe_color(image, clip_limit, tile_grid_size)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    return np.var(l)

def fitness_on_dataset_color(images, clip_limit, tile_grid_size):
    """
    Compute the average fitness over a list of color images.
    """
    scores = [fitness_function_color(img, clip_limit, tile_grid_size) 
              for img in images]
    return np.mean(scores)

# --------------------------------------------------
# 2. Particle Swarm Optimization (PSO) Components
# --------------------------------------------------
class Particle:
    def __init__(self, bounds):
        """
        Initialize a particle with a random position and zero velocity.
        The position contains [clip_limit, tile_grid_size] (tile_grid_size is later rounded).
        """
        self.position = np.array([
            random.uniform(bounds[i][0], bounds[i][1]) 
            for i in range(len(bounds))
        ], dtype=np.float32)
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_score = -np.inf

    def update_velocity(self, global_best, inertia, cognitive, social):
        r1, r2 = random.random(), random.random()
        self.velocity = (inertia * self.velocity +
                         cognitive * r1 * (self.best_position - self.position) +
                         social * r2 * (global_best - self.position))

    def update_position(self, bounds):
        self.position += self.velocity
        for i in range(len(self.position)):
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])

def mpso_color(images, bounds, num_populations=3, num_particles=10, iterations=50,
               inertia=0.7, cognitive=1.5, social=1.5, migration_interval=10):
    """
    Run Multi-Population PSO to optimize CLAHE parameters (clip_limit and tile_grid_size)
    for a list of color images.
    """
    populations = []
    for _ in range(num_populations):
        pop = [Particle(bounds) for _ in range(num_particles)]
        populations.append(pop)

    global_best = None
    global_best_score = -np.inf

    for it in range(iterations):
        for pop in populations:
            for particle in pop:
                # Use current particle position (round tile_grid_size to an integer)
                clip_limit = particle.position[0]
                tile_grid_size = int(round(particle.position[1]))
                score = fitness_on_dataset_color(images, clip_limit, tile_grid_size)
                # Update personal best
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best = particle.position.copy()
            # Update particle velocities and positions
            for particle in pop:
                particle.update_velocity(global_best, inertia, cognitive, social)
                particle.update_position(bounds)
        # Migration: share the global best among particles every few iterations
        if (it + 1) % migration_interval == 0:
            for pop in populations:
                for particle in pop:
                    particle.best_position = global_best.copy()

        print(f"Iteration {it+1}/{iterations}: Best Fitness = {global_best_score:.2f}")

    best_clip_limit = global_best[0]
    best_tile_grid_size = int(round(global_best[1]))
    return (best_clip_limit, best_tile_grid_size), global_best_score

# --------------------------------------------------
# 3. Main Script: Optimize and Convert Images
# --------------------------------------------------
def main():
    # Define folders containing the images to be converted.
    train_folder = os.path.join("train", "image")
    test_folder = os.path.join("test", "image")
    
    # ---------------------------------------------------------------------
    # Step 1: Optimize CLAHE parameters using a subset of training images.
    # ---------------------------------------------------------------------
    sample_paths = glob.glob(os.path.join(train_folder, "*.*"))
    if len(sample_paths) == 0:
        print("No images found in the train/image folder for optimization.")
        return
    
    # Use a random subset (e.g., 20 images) for parameter optimization
    num_samples = min(20, len(sample_paths))
    sample_paths = random.sample(sample_paths, num_samples)
    sample_images = []
    for path in sample_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            sample_images.append(img)
    
    bounds = [(1.0, 10.0), (4, 16)]  # (clip_limit, tile_grid_size)
    print("Optimizing CLAHE parameters using MPSO on a subset of training images...")
    best_params, best_score = mpso_color(sample_images, bounds, num_populations=3,
                                         num_particles=10, iterations=30,
                                         inertia=0.7, cognitive=1.5, social=1.5,
                                         migration_interval=10)
    best_clip_limit, best_tile_grid_size = best_params
    print(f"\nOptimized CLAHE Parameters:\n  Clip Limit = {best_clip_limit:.2f}\n  Tile Grid Size = {best_tile_grid_size}")

    # ---------------------------------------------------------------------
    # Step 2: Process and overwrite all images in train/image and test/image.
    # ---------------------------------------------------------------------
    folders = [train_folder, test_folder]
    for folder in folders:
        image_paths = glob.glob(os.path.join(folder, "*.*"))
        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue
            enhanced_img = apply_clahe_color(img, best_clip_limit, best_tile_grid_size)
            # Overwrite the original image with the enhanced version
            cv2.imwrite(img_path, enhanced_img)
            print(f"Processed {img_path}")

if __name__ == "__main__":
    main()
