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
from torch.optim.lr_scheduler import OneCycleLR
# Mixup/CutMixに必要なライブラリ
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# ==========================================
# Config: 設定
# ==========================================
class Config:
    seed = 8888 
    
    train_dir = "D:/puresotu/workspace/BreastCancer/train"
    val_dir = "D:/puresotu/workspace/BreastCancer/valid"
    test_dir = "D:/puresotu/workspace/BreastCancer/test"
    output_dir = "D:/puresotu/workspace/ts"
    
    # === モデル設定 ===
    model_name = "tf_efficientnet_b4_ns"
    img_size = 380
    
    # 【変更】Mixupを使う場合、バッチサイズが大きい方が効果が出やすいですが、
    # メモリとの兼ね合いで16のままとします（16でも十分効きます）
    batch_size = 16 
    
    # Mixupは学習が難しいので、エポック数を少し増やしても良いですが、
    # まずは40のままで「質の高い学習」を目指します
    num_epochs = 40 
    learning_rate = 4e-4 
    num_workers = 0 
    
    # Mixupのためにクラス数を2（確率分布）として扱います
    classes = ["0", "1"]
    num_classes = 2 

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
# Transforms (画像単体の拡張)
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    # RandAugmentの強度(magnitude)を少し上げて、より多様な変化を与える
    transforms.RandAugment(num_ops=2, magnitude=15), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Mixupと併用するため、Erasing確率は控えめに
    transforms.RandomErasing(p=0.1) 
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

# Test Dataset Definition (Early Definition for Loop)
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
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

# ==========================================
# 【追加】Mixup & CutMix 設定
# ==========================================
# これが今回の「transforms変更」の目玉です。
# バッチ内で画像を混ぜ合わせ、過学習を強力に防ぎます。
mixup_fn = Mixup(
    mixup_alpha=0.4,    # 画像を混ぜる度合い
    cutmix_alpha=1.0,   # 画像を切り貼りする度合い
    prob=1.0,           # 適用確率100%
    switch_prob=0.5,    # MixupとCutMixのどちらを使うか(50%)
    mode='batch', 
    label_smoothing=0.1, 
    num_classes=Config.num_classes
)

# ==========================================
# Loss Function (Mixup対応)
# ==========================================
# ラベルが「0.3, 0.7」のように混ざるため、それに対応したLoss関数を使います
criterion_train = SoftTargetCrossEntropy()
# 検証時は混ぜないので通常のCrossEntropy
criterion_val = nn.CrossEntropyLoss()

# ==========================================
# Model Definition
# ==========================================
def get_model():
    print(f"Loading {Config.model_name}...")
    model = timm.create_model(
        Config.model_name,
        pretrained=True,
        # Mixupを使うためクラス数は2 (確率分布を出力させる)
        num_classes=Config.num_classes, 
        # 【変更】Drop Path Rateを上げて、モデル内の学習をより難しくする（過学習対策）
        drop_rate=0.3,      
        drop_path_rate=0.3  # 0.25 -> 0.4 へ変更
    )
    return model

model = get_model().to(device)

# ==========================================
# Optimizer & Scheduler
# ==========================================
# 【変更】Weight Decayを増やして、パラメータの暴走（過学習）を抑える
optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=5e-2)

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
prediction_dir = os.path.join(Config.output_dir, 'Prediction')
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(prediction_dir, exist_ok=True)

save_path = os.path.join(weight_dir, 'best_model_b4_mixup.pth')

period_best_auc = 0.0
period_best_thr = 0.5 
period_best_model_path = os.path.join(weight_dir, 'temp_period_best.pth')

for epoch in range(Config.num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Train]")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 【ここが変更点】Mixup/CutMixの適用
        # inputs: 混ぜられた画像, targets: 混ぜられたラベル
        inputs, targets = mixup_fn(inputs, labels)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            # Mixup用のLoss関数を使用
            loss = criterion_train(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        loop.set_postfix(loss=loss.item())
    
    epoch_train_loss = running_loss / len(train_loader.dataset)
    
    # --- Validation (with 5-Way TTA) ---
    model.eval()
    val_running_loss = 0.0
    all_val_labels = []
    all_val_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Loss check (ValidationはMixupなしで計算)
            outputs = model(inputs)
            loss = criterion_val(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            
            # === 5-Way TTA ===
            # クラス数が2になったので、Softmaxをかけて「クラス1の確率」を取り出す
            p1 = torch.softmax(outputs, dim=1)[:, 1]
            p2 = torch.softmax(model(torch.flip(inputs, dims=[3])), dim=1)[:, 1]
            p3 = torch.softmax(model(torch.flip(inputs, dims=[2])), dim=1)[:, 1]
            p4 = torch.softmax(model(torch.rot90(inputs, k=1, dims=[2, 3])), dim=1)[:, 1]
            p5 = torch.softmax(model(torch.rot90(inputs, k=3, dims=[2, 3])), dim=1)[:, 1]
            
            probs_avg = (p1 + p2 + p3 + p4 + p5) / 5.0
            
            all_val_labels.extend(labels.cpu().numpy())
            all_val_probs.extend(probs_avg.cpu().numpy())

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    
    try:
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
    except:
        val_auc = 0.5

    print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC (5-TTA): {val_auc:.4f}")

    # --- Global Best Update ---
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), save_path)
        print(f">>> Best model saved! (AUC: {best_auc:.4f})")

    # --- Period Best Update ---
    if val_auc > period_best_auc:
        period_best_auc = val_auc
        torch.save(model.state_dict(), period_best_model_path)
        
        # 最適閾値の計算
        y_true_temp = np.array(all_val_labels)
        y_probs_temp = np.array(all_val_probs)
        best_f1_temp = 0.0
        temp_thr = 0.5
        for th in np.arange(0.01, 1.0, 0.01):
            preds_temp = (y_probs_temp >= th).astype(int)
            score_temp = f1_score(y_true_temp, preds_temp)
            if score_temp > best_f1_temp:
                best_f1_temp = score_temp
                temp_thr = th
        period_best_thr = temp_thr
        print(f"  [Period Update] Best AUC: {period_best_auc:.4f} (Thr: {period_best_thr:.2f})")

    # --- 10エポックごとにCSV作成 ---
    if (epoch + 1) % 10 == 0:
        print(f"\n=== Generating Submission for Period Epoch {epoch-8}-{epoch+1} ===")
        model.load_state_dict(torch.load(period_best_model_path))
        model.eval()
        
        predictions = []
        with torch.no_grad():
            for images, filenames in tqdm(test_loader, desc="Period Prediction"):
                images = images.to(device)
                
                # 5-Way TTA
                p1 = torch.softmax(model(images), dim=1)[:, 1]
                p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)[:, 1]
                p3 = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)[:, 1]
                p4 = torch.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)[:, 1]
                p5 = torch.softmax(model(torch.rot90(images, k=3, dims=[2, 3])), dim=1)[:, 1]
                
                probs_avg = (p1 + p2 + p3 + p4 + p5) / 5.0
                preds = (probs_avg >= period_best_thr).float().cpu().numpy().astype(int)
                
                if preds.ndim == 0: preds = [preds]
                for fn, p in zip(filenames, preds):
                    predictions.append((os.path.splitext(fn)[0], p))
        
        df = pd.DataFrame(predictions, columns=['image_id', 'label'])
        period_submit_path = os.path.join(prediction_dir, f'submission_period_epoch_{epoch+1}_auc{period_best_auc:.4f}.csv')
        df.to_csv(period_submit_path, index=False)
        print(f"Saved submission to: {period_submit_path}\n")
        period_best_auc = 0.0

# ==========================================
# Final Evaluation & Submission
# ==========================================
print("\n=== Final Evaluation with Best Model ===")
model.load_state_dict(torch.load(save_path))
model.eval()

y_true = []
y_probs = []

with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Validating (5-TTA)"):
        inputs = inputs.to(device)
        
        p1 = torch.softmax(model(inputs), dim=1)[:, 1]
        p2 = torch.softmax(model(torch.flip(inputs, dims=[3])), dim=1)[:, 1]
        p3 = torch.softmax(model(torch.flip(inputs, dims=[2])), dim=1)[:, 1]
        p4 = torch.softmax(model(torch.rot90(inputs, k=1, dims=[2, 3])), dim=1)[:, 1]
        p5 = torch.softmax(model(torch.rot90(inputs, k=3, dims=[2, 3])), dim=1)[:, 1]
        
        probs_avg = (p1 + p2 + p3 + p4 + p5) / 5.0
        
        y_true.extend(labels.numpy())
        y_probs.extend(probs_avg.cpu().numpy())

y_true = np.array(y_true)
y_probs = np.array(y_probs)

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

y_pred_final = (y_probs >= best_thr).astype(int)
cm = confusion_matrix(y_true, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Config.classes)
disp.plot(cmap=plt.cm.Blues)
plt.savefig(os.path.join(Config.output_dir, 'confusion_matrix_b4_mixup.png'))

print(f"\nGenerating Final Submission with Threshold: {best_thr:.2f}")
predictions = []

with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="Prediction"):
        images = images.to(device)
        
        p1 = torch.softmax(model(images), dim=1)[:, 1]
        p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)[:, 1]
        p3 = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)[:, 1]
        p4 = torch.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)[:, 1]
        p5 = torch.softmax(model(torch.rot90(images, k=3, dims=[2, 3])), dim=1)[:, 1]
        
        probs_avg = (p1 + p2 + p3 + p4 + p5) / 5.0
        preds = (probs_avg >= best_thr).float().cpu().numpy().astype(int)
        
        if preds.ndim == 0: preds = [preds]
        for fn, p in zip(filenames, preds):
            predictions.append((os.path.splitext(fn)[0], p))

df = pd.DataFrame(predictions, columns=['image_id', 'label'])
submit_path = os.path.join(Config.output_dir, 'Prediction', 'submission_b4_mixup.csv')
df.to_csv(submit_path, index=False)
print(f"Saved to {submit_path}")
