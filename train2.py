import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
IMG_DIR = r"C:\Users\locha\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\img_align_celeba\img_align_celeba"
IDENTITY_FILE = r"C:\Users\locha\Downloads\identity_CelebA.txt"

BATCH_SIZE = 64
EPOCHS = 20          # train longer
IMG_SIZE = 128
LR = 0.01
SPLIT_RATIO = [0.8, 0.1, 0.1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()   # ✅ AMP only if CUDA

# -------------------------
# DATASET CLASS
# -------------------------
class CelebADataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"]
        label = self.df.iloc[idx]["identity_id"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------
# MODELS
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def build_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_efficientnet(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def build_mobilenet(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# -------------------------
# TRAINING FUNCTION
# -------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, model_name="model"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val = 0
    patience, patience_counter = 5, 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = outputs.max(1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        print(f"\n[{model_name}] Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            patience_counter = 0
            print(f"✅ New best {model_name} saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping {model_name}")
                break

    return model

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    id_map = pd.read_csv(IDENTITY_FILE, sep=" ", header=None, names=["image_id", "identity_id"])
    id_map["identity_id"] -= 1
    num_classes = id_map["identity_id"].nunique()

    print(f"✅ Loaded {len(id_map)} images with {num_classes} identities")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    dataset = CelebADataset(IMG_DIR, id_map, transform)

    train_size = int(SPLIT_RATIO[0] * len(dataset))
    val_size = int(SPLIT_RATIO[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 🚀 Train multiple models
    models_to_train = {
        "SimpleCNN": SimpleCNN(num_classes),
        "ResNet18": build_resnet18(num_classes),
        "ResNet50": build_resnet50(num_classes),
        "EfficientNetB0": build_efficientnet(num_classes),
        "MobileNetV2": build_mobilenet(num_classes),
    }

    for name, model in models_to_train.items():
        print(f"\n🚀 Training {name}...")
        train_model(model, train_loader, val_loader, model_name=name)

    print("✅ Training completed for all models")