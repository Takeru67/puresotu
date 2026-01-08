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
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# ==========================================
# Config: 設定
# ==========================================
class Config:
    seed = 2024
    
    train_dir = "D:/puresotu/workspace/BreastCancer/train"
    val_dir = "D:/puresotu/workspace/BreastCancer/valid"
    test_dir = "D:/puresotu/workspace/BreastCancer/test"
    output_dir = "D:/puresotu/workspace/ts"
    
    # === モデル設定: EfficientNet-B5 (最強モデル) ===
    # 解像度を456に上げ、モデルサイズもアップして0.8超えを狙う
    model_name = "tf_efficientnet_b5_ns"
    img_size = 456
    
    # B5はメモリを使うので、2枚合計で16 (1枚あたり8) に設定
    batch_size = 16 
    
    # Mixup学習のためエポック数は多めに確保
    num_epochs = 50
    
    learning_rate = 3e-4 
    num_workers = 2 
    classes = ["0", "1"]
    num_classes = 2 

# ==========================================
# Functions & Classes
# ==========================================
def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

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

def get_model():
    print(f"Loading {Config.model_name}...")
    model = timm.create_model(
        Config.model_name,
        pretrained=True,
        num_classes=Config.num_classes, 
        drop_rate=0.3,      
        drop_path_rate=0.3 
    )
    return model

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    set_seed(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU Count: {torch.cuda.device_count()}")

    # --- Transforms (ご要望に合わせて変更) ---
    train_transforms = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        
        # 【変更点】ご指定のサイトを参考に、個別のTransformを設定
        # 1. Random Horizontal Flip (水平反転のみ)
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 2. Random Rotate (回転) - ±15度
        transforms.RandomRotation(degrees=15),
        
        # 3. Random Contrast & Brightness (ColorJitter)
        # brightness=0.2 (明度±20%), contrast=0.2 (コントラスト±20%)
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1) 
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Data Loaders ---
    train_dataset = BreastCancerDataset(Config.train_dir, Config.classes, train_transforms)
    val_dataset = BreastCancerDataset(Config.val_dir, Config.classes, val_test_transforms)
    test_dataset = TestDataset(Config.test_dir, transform=val_test_transforms)

    # WeightedRandomSampler
    targets = [label for _, label in train_dataset.data]
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, sampler=sampler, shuffle=False, num_workers=Config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    # --- Mixup & Loss ---
    mixup_fn = Mixup(
        mixup_alpha=0.4,    
        cutmix_alpha=1.0,   
        prob=1.0,           
        switch_prob=0.5,    
        mode='batch', 
        label_smoothing=0.1, 
        num_classes=Config.num_classes
    )

    criterion_train = SoftTargetCrossEntropy()
    criterion_val = nn.CrossEntropyLoss()

    # --- Model & Optimizer ---
    model = get_model().to(device)
    
    # 2枚のGPUを使う設定
    if torch.cuda.device_count() > 1:
        print(f"Activating DataParallel for {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=5e-2)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=Config.learning_rate, 
        steps_per_epoch=len(train_loader), 
        epochs=Config.num_epochs
    )
    scaler = GradScaler()

    # --- Training Loop ---
    best_auc = 0.0
    weight_dir = os.path.join(Config.output_dir, 'Weight_B5')
    prediction_dir = os.path.join(Config.output_dir, 'Prediction_B5')
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    save_path = os.path.join(weight_dir, 'best_model_b5_2gpu.pth')
    
    period_best_auc = 0.0
    period_best_thr = 0.5 
    period_best_model_path = os.path.join(weight_dir, 'temp_period_best.pth')
    
    saved_model_paths = []
    period_start = 1

    for epoch in range(Config.num_epochs):
        if epoch % 10 == 0:
            period_best_auc = 0.0
            period_start = epoch + 1
            
        # Training
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Train]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            inputs, targets = mixup_fn(inputs, labels)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion_train(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            loop.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion_val(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                
                # 5-Way TTA
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

        # Save Global Best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f">>> Global Best saved! (AUC: {best_auc:.4f})")
            if save_path not in saved_model_paths:
                saved_model_paths.append(save_path)

        # Save Period Best
        if val_auc > period_best_auc:
            period_best_auc = val_auc
            period_model_name = f'best_model_epoch_{period_start}-{min(period_start+9, Config.num_epochs)}.pth'
            period_model_path = os.path.join(weight_dir, period_model_name)
            torch.save(model.state_dict(), period_model_path)
            
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
            print(f"  [Period Best] Saved: {period_model_name} (AUC: {period_best_auc:.4f}, Thr: {period_best_thr:.2f})")
            
            if period_model_path not in saved_model_paths:
                saved_model_paths.append(period_model_path)

        # 10エポックごとにCSV作成
        if (epoch + 1) % 10 == 0:
            print(f"\n=== Generating Submission for Period Epoch {epoch-8}-{epoch+1} ===")
            temp_model = timm.create_model(Config.model_name, pretrained=False, num_classes=Config.num_classes)
            temp_model = temp_model.to(device)
            if torch.cuda.device_count() > 1:
                temp_model = nn.DataParallel(temp_model)
                
            temp_model.load_state_dict(torch.load(period_model_path))
            temp_model.eval()
            
            predictions = []
            with torch.no_grad():
                for images, filenames in tqdm(test_loader, desc="Period Prediction"):
                    images = images.to(device)
                    
                    p1 = torch.softmax(temp_model(images), dim=1)[:, 1]
                    p2 = torch.softmax(temp_model(torch.flip(images, dims=[3])), dim=1)[:, 1]
                    p3 = torch.softmax(temp_model(torch.flip(images, dims=[2])), dim=1)[:, 1]
                    p4 = torch.softmax(temp_model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)[:, 1]
                    p5 = torch.softmax(temp_model(torch.rot90(images, k=3, dims=[2, 3])), dim=1)[:, 1]
                    
                    probs_avg = (p1 + p2 + p3 + p4 + p5) / 5.0
                    preds = (probs_avg >= period_best_thr).float().cpu().numpy().astype(int)
                    
                    if preds.ndim == 0: preds = [preds]
                    for fn, p in zip(filenames, preds):
                        predictions.append((os.path.splitext(fn)[0], p))
            
            df = pd.DataFrame(predictions, columns=['image_id', 'label'])
            period_submit_path = os.path.join(prediction_dir, f'submission_period_epoch_{epoch+1}_auc{period_best_auc:.4f}.csv')
            df.to_csv(period_submit_path, index=False)
            print(f"Saved submission to: {period_submit_path}\n")
            
            del temp_model
            torch.cuda.empty_cache()

    # --- Final Ensemble ---
    print("\n=== Starting Final Ensemble Prediction ===")
    print(f"Models for ensemble: {[os.path.basename(p) for p in saved_model_paths]}")

    saved_model_paths = list(set(saved_model_paths))
    final_ensemble_preds = None
    test_filenames = []

    temp_model = timm.create_model(Config.model_name, pretrained=False, num_classes=Config.num_classes)
    temp_model = temp_model.to(device)
    if torch.cuda.device_count() > 1:
        temp_model = nn.DataParallel(temp_model)

    for model_path in saved_model_paths:
        if not os.path.exists(model_path): continue
        
        print(f"Ensembling: {os.path.basename(model_path)}...")
        temp_model.load_state_dict(torch.load(model_path))
        temp_model.eval()
        
        temp_preds = []
        collect_names = (len(test_filenames) == 0)
        
        with torch.no_grad():
            for images, filenames in tqdm(test_loader, desc="Ensemble Predict"):
                images = images.to(device)
                
                p1 = torch.softmax(temp_model(images), dim=1)[:, 1]
                p2 = torch.softmax(temp_model(torch.flip(images, dims=[3])), dim=1)[:, 1]
                p3 = torch.softmax(temp_model(torch.flip(images, dims=[2])), dim=1)[:, 1]
                p4 = torch.softmax(temp_model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)[:, 1]
                p5 = torch.softmax(temp_model(torch.rot90(images, k=3, dims=[2, 3])), dim=1)[:, 1]
                
                probs_avg = (p1 + p2 + p3 + p4 + p5) / 5.0
                temp_preds.extend(probs_avg.cpu().numpy())
                
                if collect_names:
                    test_filenames.extend(filenames)
        
        if final_ensemble_preds is None:
            final_ensemble_preds = np.array(temp_preds)
        else:
            final_ensemble_preds += np.array(temp_preds)

    final_ensemble_preds /= len(saved_model_paths)

    ensemble_thr = 0.5
    binary_preds = (final_ensemble_preds >= ensemble_thr).astype(int)

    predictions = []
    for fn, p in zip(test_filenames, binary_preds):
        predictions.append((os.path.splitext(fn)[0], p))

    df = pd.DataFrame(predictions, columns=['image_id', 'label'])
    submit_path = os.path.join(Config.output_dir, 'Prediction_B5', 'submission_ensemble_final_b5.csv')
    df.to_csv(submit_path, index=False)
    print(f"\nFinal Ensemble Submission Saved to: {submit_path}")
    print("Congratulations! The ensemble of all best models is complete.")
