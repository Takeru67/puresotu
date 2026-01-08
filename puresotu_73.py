import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from torch.cuda.amp import autocast, GradScaler
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# ==========================================
# Config: 設定
# ==========================================
class Config:
    seed = 1234
    
    # パス設定
    train_dir = "D:/puresotu/workspace/BreastCancer/train"
    val_dir = "D:/puresotu/workspace/BreastCancer/valid"
    test_dir = "D:/puresotu/workspace/BreastCancer/test"
    output_dir = "D:/puresotu/workspace/ts"
    
    # === モデル設定 (EfficientNet-B3) ===
    model_name = "efficientnet_b3"
    img_size = 300   # B3推奨サイズ
    batch_size = 16  # メモリ溢れ注意
    
    # 【変更】回数を増やして最後まで学習させる
    num_epochs = 50
    learning_rate = 1e-4
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
    transforms.TrivialAugmentWide(), # 強力なデータ拡張
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
# Data Loaders (WeightedRandomSampler)
# ==========================================
train_dataset = BreastCancerDataset(Config.train_dir, Config.classes, train_transforms)
val_dataset = BreastCancerDataset(Config.val_dir, Config.classes, val_test_transforms)

# 不均衡データ対策
targets = [label for _, label in train_dataset.data]
class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in targets])
samples_weight = torch.from_numpy(samples_weight)

sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, sampler=sampler, shuffle=False, num_workers=Config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

print("WeightedRandomSampler applied.")

# ==========================================
# Model Definition
# ==========================================
def get_model():
    model = timm.create_model(
        Config.model_name,
        pretrained=True,
        num_classes=2,
        drop_rate=0.3,
        drop_path_rate=0.2
    )
    return model

model = get_model().to(device)

# ==========================================
# Mixup, Loss, Optimizer
# ==========================================
# 【変更】難易度調整
mixup_fn = Mixup(
    mixup_alpha=0.4,    # 0.8 -> 0.4 に緩和
    cutmix_alpha=0.0,   # CutMixは無効化 (病変部を消してしまうリスク回避)
    prob=1.0, 
    switch_prob=0.0,    # CutMixを使わないので0
    mode='batch', 
    label_smoothing=0.1, 
    num_classes=2
)

criterion_train = SoftTargetCrossEntropy()
criterion_val = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs, eta_min=1e-6)
scaler = GradScaler()

# ==========================================
# Training Loop
# ==========================================
best_auc = 0.0
weight_dir = os.path.join(Config.output_dir, 'Weight')
os.makedirs(weight_dir, exist_ok=True)
save_path = os.path.join(weight_dir, 'best_model_b3_tta.pth')

for epoch in range(Config.num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Train]")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Mixup
        inputs, targets = mixup_fn(inputs, labels)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion_train(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        loop.set_postfix(loss=loss.item())
    
    epoch_train_loss = running_loss / len(train_loader.dataset)
    
    # --- Validation (with TTA) ---
    # TTA: 通常画像と反転画像の予測を平均する
    model.eval()
    val_running_loss = 0.0
    all_val_labels = []
    all_val_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. 通常予測
            outputs = model(inputs)
            loss = criterion_val(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            probs1 = torch.softmax(outputs, dim=1)[:, 1]

            # 2. 反転画像予測 (TTA)
            inputs_flipped = torch.flip(inputs, dims=[3]) # 水平反転
            outputs_flipped = model(inputs_flipped)
            probs2 = torch.softmax(outputs_flipped, dim=1)[:, 1]
            
            # 平均
            probs_avg = (probs1 + probs2) / 2.0
            
            all_val_labels.extend(labels.cpu().numpy())
            all_val_probs.extend(probs_avg.cpu().numpy())

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    
    try:
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
    except:
        val_auc = 0.5

    print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC (TTA): {val_auc:.4f}")
    
    scheduler.step()

    # Best Model Save
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), save_path)
        print(f">>> Best model saved! (AUC: {best_auc:.4f})")

# ==========================================
# Evaluation & Threshold Optimization (with TTA)
# ==========================================
print("\n=== Best Model Evaluation & Threshold Optimization ===")
model.load_state_dict(torch.load(save_path))
model.eval()

y_true = []
y_probs = []

with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Validating (TTA)"):
        inputs = inputs.to(device)
        
        # TTA
        outputs1 = model(inputs)
        outputs2 = model(torch.flip(inputs, dims=[3]))
        
        probs1 = torch.softmax(outputs1, dim=1)[:, 1]
        probs2 = torch.softmax(outputs2, dim=1)[:, 1]
        probs_avg = (probs1 + probs2) / 2.0
        
        y_true.extend(labels.numpy())
        y_probs.extend(probs_avg.cpu().numpy())

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# 最適閾値
best_thr = 0.5
best_f1 = 0.0
thresholds_search = np.arange(0.01, 1.0, 0.01)

for th in thresholds_search:
    preds = (y_probs >= th).astype(int)
    score = f1_score(y_true, preds)
    if score > best_f1:
        best_f1 = score
        best_thr = th

print(f"\nBest Threshold: {best_thr:.2f}")
print(f"Best F1 Score : {best_f1:.4f}")
print("AUC:", roc_auc_score(y_true, y_probs))

# 混同行列
y_pred_final = (y_probs >= best_thr).astype(int)
cm = confusion_matrix(y_true, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Config.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Thr={best_thr:.2f})")
plt.savefig(os.path.join(Config.output_dir, 'confusion_matrix_b3_tta.png'))
print("Confusion Matrix saved.")

# ==========================================
# Submission (with TTA)
# ==========================================
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
print(f"\nGenerating submission with Threshold: {best_thr:.2f} & TTA")

with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="Prediction (TTA)"):
        images = images.to(device)
        
        # TTA
        outputs1 = model(images)
        outputs2 = model(torch.flip(images, dims=[3]))
        
        probs1 = torch.softmax(outputs1, dim=1)[:, 1]
        probs2 = torch.softmax(outputs2, dim=1)[:, 1]
        probs_avg = (probs1 + probs2) / 2.0
        
        preds = (probs_avg >= best_thr).float().cpu().numpy().astype(int)
        
        if preds.ndim == 0: preds = [preds]
            
        for fn, p in zip(filenames, preds):
            image_id = os.path.splitext(fn)[0]
            predictions.append((image_id, p))

prediction_dir = os.path.join(Config.output_dir, 'Prediction')
os.makedirs(prediction_dir, exist_ok=True)
submit_file_path = f'{prediction_dir}/submission.csv'
df = pd.DataFrame(predictions, columns=['image_id', 'label'])
df.to_csv(submit_file_path, index=False)
print(f"Saved to {submit_file_path}")
