import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from carvana_dataset import CarvanaDataset

# Hyperparameters
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
EPOCHS = 2000
DATA_PATH = "data"
MODEL_SAVE_PATH = "models/unet.pth"
TRAIN = True  # Set this to False for testing

# Device configuration
device = "cpu"

# Initialize dataset
if TRAIN:
    # Load training dataset and split into training and validation sets
    full_dataset = CarvanaDataset(DATA_PATH, test=False)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Data loaders for training and validation
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
else:
    # Load test dataset
    test_dataset = CarvanaDataset(DATA_PATH, test=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, optimizer, and loss function
model = UNet(in_channels=3, num_classes=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

# Training and validation loop
if TRAIN:
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        
        # Training phase
        model.train()
        train_running_loss = 0
        for img, mask in tqdm(train_dataloader, desc="Training", leave=False):
            img = img.to(device)
            mask = mask.to(device)

            # Forward pass
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()

        train_loss = train_running_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for img, mask in tqdm(val_dataloader, desc="Validation", leave=False):
                img = img.to(device)
                mask = mask.to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                
                val_running_loss += loss.item()
            
            val_loss = val_running_loss / len(val_dataloader)
        
        # Logging the losses
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    # Save the model after training
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# Testing phase
else:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    print("Testing Model...")

    with torch.no_grad():
        for idx, img in enumerate(tqdm(test_dataloader, desc="Testing")):
            img = img.to(device)
            
            # Forward pass
            y_pred = model(img)
            
            # Save or process predictions here if needed
            # For example, converting predictions to binary masks, resizing, saving, etc.
            # Example placeholder: pred_mask = (torch.sigmoid(y_pred) > 0.5).float()
    
    print("Testing complete.")
