import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        self.test = test
        
        if test:
            # Test images (assuming no labels/masks in test set)
            test_img_path = os.path.join(root_path, "test")
            if not os.path.exists(test_img_path):
                raise FileNotFoundError(f"Directory {test_img_path} does not exist.")
            
            self.images = sorted([test_img_path + "/" + i for i in os.listdir(test_img_path)])
            self.masks = None  # No masks for test images
        else:
            # Train images and labels
            train_img_path = os.path.join(root_path, "train", "image")
            train_mask_path = os.path.join(root_path, "train", "label")
            
            # Check if paths exist
            if not os.path.exists(train_img_path) or not os.path.exists(train_mask_path):
                raise FileNotFoundError(f"Directories {train_img_path} or {train_mask_path} do not exist.")
            
            self.images = sorted([train_img_path + "/" + i for i in os.listdir(train_img_path)])
            self.masks = sorted([train_mask_path + "/" + i for i in os.listdir(train_mask_path)])
        
        # Transformation to resize and convert images/masks to tensors
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        img = self.transform(img)
        
        if self.test:
            return img  # Only return the image in test mode
        
        mask = Image.open(self.masks[index]).convert("L")
        mask = self.transform(mask)
        return img, mask

    def __len__(self):
        return len(self.images)
