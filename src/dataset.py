import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import cv2
import numpy as np

# Increase PIL pixel limit to handle large dermoscopic images safely
Image.MAX_IMAGE_PIXELS = 300_000_000

# Class map
CLASS_TO_IDX = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

class HairRemoval(object):
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, img):
        # Resize to target size first to optimize inpainting speed
        img_resized = img.resize(self.size)
        img_np = np.array(img_resized)
        
        # DullRazor algorithm
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Morphological blackhat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Thresholding
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpaint
        inpainted = cv2.inpaint(img_np, thresh, 1, cv2.INPAINT_TELEA)
        
        return Image.fromarray(inpainted)

class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = row['label_idx']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return a black image if file is corrupt or unreadable
            print(f"Warning: Could not load image {img_path}: {e}. Using blank image.")
            image = Image.new('RGB', (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(remove_hair=False):
    train_ops = []
    val_test_ops = []
    
    if remove_hair:
        # Prepend hair removal to the transform list
        train_ops.append(HairRemoval((224, 224)))
        val_test_ops.append(HairRemoval((224, 224)))
        
        # Crop/Augment
        train_ops.extend([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
        ])
        
        val_test_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_ops.extend([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
        ])
        
        val_test_ops.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    train_transform = transforms.Compose(train_ops)
    val_test_transform = transforms.Compose(val_test_ops)
    
    return train_transform, val_test_transform

def get_dataloaders(splits_dir='data/splits', batch_size=32, use_weighted_sampler=False, sampler_beta=0.35, remove_hair=False):
    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'dev.csv'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))

    train_transform, val_test_transform = get_transforms(remove_hair=remove_hair)

    train_dataset = HAM10000Dataset(train_df, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, transform=val_test_transform)
    test_dataset = HAM10000Dataset(test_df, transform=val_test_transform)

    print(f"Train samples: {len(train_dataset)}, Val (dev): {len(val_dataset)}, Test: {len(test_dataset)}")
    
    sampler = None
    if use_weighted_sampler:
        # Calculate class weights for sampler
        class_counts = train_df['label_idx'].value_counts().sort_index()
        # Smooth inverse frequency oversampling to prevent aggressive overfitting
        class_weights = 1.0 / (class_counts ** sampler_beta)
        # Map class weights to each sample in training set
        sample_weights = train_df['label_idx'].map(class_weights).values
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    num_workers = int(os.environ.get('NUM_WORKERS', 0))
    # If using sampler, shuffle must be False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler,
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df

