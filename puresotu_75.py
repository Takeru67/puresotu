import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from torch.cuda.amp import autocast, GradScaler
import timm
# CosineAnnealingWarmRestarts から OneCycleLR に変更（収束力アップ）
from torch.optim.lr_scheduler import OneCycleLR

# ==========================================
# Config: 設定
# ==========================================
class Config:
    seed = 2024 # 気分を変えて新しいシード値
    
    # パス設定 (環境に合わせて変更してください)
    train_dir = "D:/puresotu/workspace/BreastCancer/train"
    val_dir = "D:/puresotu/workspace/BreastCancer/valid"
    test_dir = "D:/puresotu/workspace/BreastCancer/test"
    output_dir = "D:/puresotu/workspace/ts"
    
    # === モデル設定 (EfficientNet-B4 + 高解像度) ===
    # B3(300px) -> B4(380px) へ強化
    model_name = "tf_efficientnet_b4_ns"
    img_size = 380
    
    # 画像サイズを上げたのでバッチサイズを下げてOOM回避
    batch_size = 16 
    
    num_epochs = 40 # OneCycleLR用に少し短縮して集中学習
    learning_rate = 5e-4 # OneCycleLRの最大学習率
    num_workers = 0 
    classes = ["0", "1"]

# ==========================================
# Seed Setting
# ==========================================
def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(Config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# Dataset
# ==========================================
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.data = []
        self._prepare_data()

    def _prepare_data(self):
        for class_label in self.classes:
            class_path = os.path.join(self.root_dir, class_label)
            if not os.path.isdir(class_path):
                continue
            label_index = self.classes.index(class_label)
            for img_file in os.listdir(class_path):
                img_full_path = os.path.join(class_path, img_file)
                if os.path.isfile(img_full_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append((img_full_path, label_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (Config.img_size, Config.img_size))
        
        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================================
# Transforms
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    # 医用画像は上下左右の関係がないことが多いのでVerticalFlipも重要
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5), 
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 色調変化への耐性
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# Data Loaders
# ==========================================
train_dataset = BreastCancerDataset(Config.train_dir, Config.classes, train_transforms)
val_dataset = BreastCancerDataset(Config.val_dir, Config.classes, val_test_transforms)

# WeightedRandomSampler
targets = [label for _, label in train_dataset.data]
class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in targets])
samples_weight = torch.from_numpy(samples_weight)

sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, sampler=sampler, shuffle=False, num_workers=Config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

# ==========================================
# Loss Function: Focal Loss
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# ==========================================
# Model Definition
# ==========================================
def get_model():
    print(f"Loading {Config.model_name}...")
    model = timm.create_model(
        Config.model_name,
        pretrained=True,
        num_classes=1,
        drop_rate=0.4,
        drop_path_rate=0.2
    )
    return model

model = get_model().to(device)

# ==========================================
# Optimizer & Scheduler
# ==========================================
criterion = FocalLoss(alpha=1, gamma=2.0).to(device)
optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=1e-2)

# OneCycleLR: 学習率を山なりに変化させ、最後に急激に下げる（過学習を防ぎつつ精度を出す）
scheduler = OneCycleLR(
    optimizer, 
    max_lr=Config.learning_rate, 
    steps_per_epoch=len(train_loader), 
    epochs=Config.num_epochs
)

scaler = GradScaler()

# ==========================================
# Training Loop
# ==========================================
best_auc = 0.0
weight_dir = os.path.join(Config.output_dir, 'Weight')
os.makedirs(weight_dir, exist_ok=True)
save_path = os.path.join(weight_dir, 'best_model_b4_380.pth')

for epoch in range(Config.num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Train]")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # OneCycleLRはbatchごとにstepする
        
        running_loss += loss.item() * inputs.size(0)
        loop.set_postfix(loss=loss.item())
    
    epoch_train_loss = running_loss / len(train_loader.dataset)
    
    # --- Validation (with 3-Way TTA) ---
    model.eval()
    val_running_loss = 0.0
    all_val_labels = []
    all_val_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels_unsqueezed = labels.unsqueeze(1)
            
            # Loss計算用（通常画像のみ）
            outputs = model(inputs)
            loss = criterion(outputs, labels_unsqueezed)
            val_running_loss += loss.item() * inputs.size(0)
            
            # === 3-Way TTA (通常 + 水平反転 + 垂直反転) ===
            # 1. Normal
            probs1 = torch.sigmoid(outputs).view(-1)
            
            # 2. Horizontal Flip
            inputs_h = torch.flip(inputs, dims=[3])
            probs2 = torch.sigmoid(model(inputs_h)).view(-1)
            
            # 3. Vertical Flip (追加!)
            inputs_v = torch.flip(inputs, dims=[2])
            probs3 = torch.sigmoid(model(inputs_v)).view(-1)

            # 平均を取る
            probs_avg = (probs1 + probs2 + probs3) / 3.0
            
            all_val_labels.extend(labels.cpu().numpy())
            all_val_probs.extend(probs_avg.cpu().numpy())

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    
    try:
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
    except:
        val_auc = 0.5

    print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC (3-TTA): {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), save_path)
        print(f">>> Best model saved! (AUC: {best_auc:.4f})")

# ==========================================
# Evaluation & Submission (with 3-Way TTA)
# ==========================================
print("\n=== Final Evaluation with Best Model ===")
model.load_state_dict(torch.load(save_path))
model.eval()

y_true = []
y_probs = []

with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Validating (3-TTA)"):
        inputs = inputs.to(device)
        
        # 3-Way TTA
        p1 = torch.sigmoid(model(inputs)).view(-1)
        p2 = torch.sigmoid(model(torch.flip(inputs, dims=[3]))).view(-1) # H-Flip
        p3 = torch.sigmoid(model(torch.flip(inputs, dims=[2]))).view(-1) # V-Flip
        
        probs_avg = (p1 + p2 + p3) / 3.0
        
        y_true.extend(labels.numpy())
        y_probs.extend(probs_avg.cpu().numpy())

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# Best Threshold Search
best_thr = 0.5
best_f1 = 0.0
for th in np.arange(0.01, 1.0, 0.01):
    preds = (y_probs >= th).astype(int)
    score = f1_score(y_true, preds)
    if score > best_f1:
        best_f1 = score
        best_thr = th

print(f"\nBest Threshold: {best_thr:.2f}")
print(f"Best F1 Score : {best_f1:.4f}")
print("AUC:", roc_auc_score(y_true, y_probs))

# Save Confusion Matrix
y_pred_final = (y_probs >= best_thr).astype(int)
cm = confusion_matrix(y_true, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Config.classes)
disp.plot(cmap=plt.cm.Blues)
plt.savefig(os.path.join(Config.output_dir, 'confusion_matrix_b4_380.png'))

# --- Submission ---
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_file

test_dataset = TestDataset(Config.test_dir, transform=val_test_transforms)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

predictions = []
print(f"\nGenerating submission with Threshold: {best_thr:.2f} & 3-Way TTA")

with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="Prediction"):
        images = images.to(device)
        
        # 3-Way TTA
        p1 = torch.sigmoid(model(images)).view(-1)
        p2 = torch.sigmoid(model(torch.flip(images, dims=[3]))).view(-1)
        p3 = torch.sigmoid(model(torch.flip(images, dims=[2]))).view(-1)
        
        probs_avg = (p1 + p2 + p3) / 3.0
        preds = (probs_avg >= best_thr).float().cpu().numpy().astype(int)
        
        if preds.ndim == 0: preds = [preds]
        for fn, p in zip(filenames, preds):
            predictions.append((os.path.splitext(fn)[0], p))

df = pd.DataFrame(predictions, columns=['image_id', 'label'])
submit_path = os.path.join(Config.output_dir, 'Prediction', 'submission_b4_380.csv')
os.makedirs(os.path.dirname(submit_path), exist_ok=True)
df.to_csv(submit_path, index=False)
print(f"Saved to {submit_path}")
