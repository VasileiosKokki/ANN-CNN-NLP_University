from torch.utils.data import Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

class COVID19Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized by class subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']  # The class labels
        self.image_paths = []
        self.labels = []

        # Traverse directories and store image paths and labels
        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_folder):
                self.image_paths.append(os.path.join(class_folder, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve
        Returns:
            sample (dict): A dictionary with 'image' and 'label'
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label

    def display_batch(self, indexes):
        """
        Displays a batch of images with corresponding labels.

        Args:
            indexes (list): List of indexes of images to display.
        """
        images = []
        labels = []

        # Load images and labels from the indexes
        for idx in indexes:
            image, label = self[idx]
            images.append(image)
            labels.append(self.classes[label])

        # Create a grid of subfigures
        n = len(images)
        rows = int(n**0.5)  # Square root to create roughly a square grid
        cols = (n + rows - 1) // rows  # Handle cases when n is not a perfect square

        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        axes = axes.ravel()  # Flatten to make indexing easier

        for i in range(n):
            axes[i].imshow(images[i].permute(1, 2, 0))  # Permute to HWC format for displaying
            axes[i].set_title(labels[i])
            axes[i].axis('off')  # Turn off axis

        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].axis('off')

        plt.show()