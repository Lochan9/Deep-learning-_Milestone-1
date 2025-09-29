import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------
# CONFIG
# -------------------------
IMG_DIR = r"C:\Users\locha\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\img_align_celeba\img_align_celeba"
IDENTITY_FILE = r"C:\Users\locha\Downloads\identity_CelebA.txt"
BATCH_SIZE = 64
IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def build_resnet(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -------------------------
# LOAD DATA
# -------------------------
id_map = pd.read_csv(IDENTITY_FILE, sep=" ", header=None, names=["image_id", "identity_id"])
id_map["identity_id"] -= 1
num_classes = id_map["identity_id"].nunique()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = CelebADataset(IMG_DIR, id_map, transform)

# Proper split sizes
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -------------------------
# LOAD TRAINED MODELS
# -------------------------
simple_model = SimpleCNN(num_classes).to(device)
simple_model.load_state_dict(torch.load("SimpleCNN_best.pth", map_location=device))
simple_model.eval()

resnet_model = build_resnet(num_classes).to(device)
resnet_model.load_state_dict(torch.load("ResNet18_best.pth", map_location=device))
resnet_model.eval()

# -------------------------
# EVALUATION FUNCTION
# -------------------------
def evaluate_model(model, loader, model_name):
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\n🔹 {model_name} Performance on Test Set")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return acc, prec, rec, f1

# -------------------------
# RUN EVALUATION
# -------------------------
results = {}
for model, name in [(simple_model, "SimpleCNN"), (resnet_model, "ResNet18")]:
    results[name] = evaluate_model(model, test_loader, name)

# -------------------------
# VISUALIZE COMPARISON
# -------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = range(len(metrics))

plt.figure(figsize=(8, 5))
for i, (name, scores) in enumerate(results.items()):
    plt.plot(x, scores, marker="o", label=name)

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Model Comparison on CelebA Test Set")
plt.legend()
plt.grid(True)
plt.show()
