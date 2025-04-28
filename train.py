import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")

    train_csv = r"C:\Users\STIC-11\Desktop\Sk2\train.csv"
    val_csv = r"C:\Users\STIC-11\Desktop\Sk2\val.csv"
    image_dir = r"C:\Users\STIC-11\Desktop\Sk2\data\train"

    for file in [train_csv, val_csv]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"CSV file not found: {file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError("Image directory not found.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5353487730026245, 0.36283448338508606, 0.2486359179019928], 
                             std=[0.21264150738716125, 0.158598393201828, 0.14018195867538452])
    ])

    class RetinopathyDataset(Dataset):
        def __init__(self, csv_file, root_dir, transform=None):
            self.data = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform
            self.valid_indices = []

            print("Validating dataset files...")
            for idx in tqdm(range(len(self.data))):
                filename = str(self.data.iloc[idx, 0])
                if not filename.endswith(".jpeg"):
                    filename += ".jpeg"
                img_path = os.path.join(self.root_dir, filename)

                if os.path.exists(img_path):
                    self.valid_indices.append(idx)

            print(f"Found {len(self.valid_indices)} valid images out of {len(self.data)} entries")
            self.data = self.data.iloc[self.valid_indices].reset_index(drop=True)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            filename = str(self.data.iloc[idx, 0])
            if not filename.endswith(".jpeg"):
                filename += ".jpeg"
            img_path = os.path.join(self.root_dir, filename)

            try:
                image = Image.open(img_path).convert("RGB")
                label = int(self.data.iloc[idx, 1])

                if self.transform:
                    image = self.transform(image)

                return image, label
            except Exception as e:
                print(f" Skipping corrupted image: {img_path}, Error: {e}")
                return None, None

    print("Initializing datasets...")
    train_dataset = RetinopathyDataset(csv_file=train_csv, root_dir=image_dir, transform=transform)
    val_dataset = RetinopathyDataset(csv_file=val_csv, root_dir=image_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(1024, 5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scaler = GradScaler()

    num_epochs = 50
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, leave=True)
        for inputs, labels in loop:
            if inputs is None:
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"): 
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), "checkpoints/best_model.ckpt")
            print(f" Best model updated with val_acc: {epoch_acc:.2f}%")

    print(" Training completed!")
