import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import warnings


warnings.filterwarnings('ignore')

import os
os.makedirs('charts', exist_ok=True)

class UnifiedDefectDataset(Dataset):
    """Unified dataset for defect detection across multiple datasets"""
    
    def __init__(self, root_dirs: Dict[str, str], 
                 split: str = 'train', transform=None, 
                 include_datasets: List[str] = None):
        """
        Args:
            root_dirs: Dictionary mapping dataset names to their root directories
                      e.g., {'DAGM': 'path/to/dagm', 'MT': 'path/to/mt', 'PCB': 'path/to/pcb'}
            split: 'train' or 'test'
            transform: Data transforms
            include_datasets: List of datasets to include, None for all
        """
        self.root_dirs = root_dirs
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.dataset_info = {}
        
        # Define dataset configurations
        self.dataset_configs = {
            'DAGM': {
                'classes': [f'DAGM_Class{i}' for i in range(1, 11)],
                'load_method': self._load_dagm,
                'description': 'DAGM 2007 Competition Dataset'
            },
            'MT': {
                'classes': ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Free', 'MT_Uneven'],
                'load_method': self._load_magnetic_tile,
                'description': 'Magnetic Tile Defect Dataset'
            }
        }
        
        # Filter datasets if specified
        if include_datasets:
            self.dataset_configs = {k: v for k, v in self.dataset_configs.items() 
                                  if k in include_datasets}
        
        self._build_unified_class_mapping()
        self._load_all_datasets()
        if self.split == 'train':
            self._upsample_mt_classes()

    def _upsample_mt_classes(self):
        """Artificially upsample MT classes to balance within MT only"""
        from collections import defaultdict
        import random

        if 'MT' not in self.dataset_configs:
            return  # No MT dataset to balance

        print("\nUpsampling MT classes to balance within MT...")

        class_sample_map = defaultdict(list)

        # Separate MT samples by class
        for sample in self.samples:
            if sample[2] == 'MT':  # sample[2] is dataset_name
                class_sample_map[sample[1]].append(sample)  # sample[1] is label_idx

        # Find target (max) class size
        max_count = max(len(v) for v in class_sample_map.values())

        # Create new balanced sample list
        new_samples = []
        for cls, samples in class_sample_map.items():
            if len(samples) < max_count:
                replicated = samples * (max_count // len(samples))
                remainder = random.choices(samples, k=max_count - len(replicated))
                upsampled = replicated + remainder
            else:
                upsampled = samples

            new_samples.extend(upsampled)

        # Keep non-MT samples unchanged
        non_mt_samples = [s for s in self.samples if s[2] != 'MT']
        self.samples = non_mt_samples + new_samples
        print(f"MT classes upsampled to {max_count} samples each.")

        
    def _build_unified_class_mapping(self):
        """Build unified class mapping across all datasets"""
        all_classes = []
        dataset_class_ranges = {}
        
        current_idx = 0
        for dataset_name, config in self.dataset_configs.items():
            classes = config['classes']
            dataset_class_ranges[dataset_name] = (current_idx, current_idx + len(classes))
            all_classes.extend(classes)
            current_idx += len(classes)
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.dataset_class_ranges = dataset_class_ranges
        
        print(f"Unified class mapping created with {len(all_classes)} total classes:")
        for dataset_name, (start, end) in dataset_class_ranges.items():
            print(f"  {dataset_name}: classes {start}-{end-1} ({end-start} classes)")
    
    def _load_all_datasets(self):
        """Load all specified datasets"""
        for dataset_name, config in self.dataset_configs.items():
            if dataset_name in self.root_dirs:
                print(f"Loading {dataset_name} dataset...")
                samples_before = len(self.samples)
                config['load_method'](dataset_name, self.root_dirs[dataset_name])
                samples_after = len(self.samples)
                
                self.dataset_info[dataset_name] = {
                    'samples_count': samples_after - samples_before,
                    'classes': config['classes'],
                    'description': config['description']
                }
                
                print(f"  Loaded {samples_after - samples_before} samples from {dataset_name}")
    
    def _load_dagm(self, dataset_name: str, root_dir: str):
        """Load DAGM dataset"""
        original_classes = [f'Class{i}' for i in range(1, 11)]
        unified_classes = [f'DAGM_Class{i}' for i in range(1, 11)]
        
        for orig_class, unified_class in zip(original_classes, unified_classes):
            class_dir = os.path.join(root_dir, orig_class, 
                                   'Train' if self.split == 'train' else 'Test')
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(('.PNG', '.jpg', '.jpeg')) and not img_file.endswith('_label.PNG'):
                        img_path = os.path.join(class_dir, img_file)
                        label_idx = self.class_to_idx[unified_class]
                        self.samples.append((img_path, label_idx, dataset_name, unified_class))
    
    def _load_magnetic_tile(self, dataset_name: str, root_dir: str):
        """Load Magnetic Tile dataset"""
        classes = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Free', 'MT_Uneven']
        
        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name, 'Imgs')
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(class_dir, img_file)
                        label_idx = self.class_to_idx[class_name]
                        self.samples.append((img_path, label_idx, dataset_name, class_name))
    
    def get_dataset_statistics(self):
        """Get statistics about the combined dataset"""
        stats = {
            'total_samples': len(self.samples),
            'total_classes': len(self.class_to_idx),
            'datasets': self.dataset_info
        }
        
        # Class distribution
        class_counts = {}
        dataset_counts = {}
        
        for _, label_idx, dataset_name, class_name in self.samples:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        
        stats['class_distribution'] = class_counts
        stats['dataset_distribution'] = dataset_counts
        
        return stats
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, dataset_name, class_name = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, {'path': img_path, 'dataset': dataset_name, 'class': class_name}

class EfficientNetDefectClassifier(nn.Module):
    """Enhanced EfficientNet-B0 based defect classifier for multiple datasets"""
    
    def __init__(self, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.3):
        super(EfficientNetDefectClassifier, self).__init__()

        # Load EfficientNet-B0 backbone
        self.backbone = efficientnet_b0(pretrained=pretrained)

        # Replace classifier with a simple linear layer for our target classes
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(num_features, num_classes)

        # Store features for Grad-CAM
        self.features = None
        self.backbone.features.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        self.features = output

    def forward(self, x):
        return self.backbone(x)

class GradCAM:
    """Grad-CAM implementation for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        output = self.model(input_tensor)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        class_score = output[:, class_idx].sum()
        class_score.backward(retain_graph=True)

        # Generate CAM
        gradients = self.gradients[0]         # (C, H, W)
        activations = self.activations[0]     # (C, H, W)

        # Ensure gradients and activations are on same device
        device = activations.device
        gradients = gradients.to(device)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Make sure weights and activations are on same device
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = torch.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

class UnifiedDefectDetectionTrainer:
    """Main trainer class for unified defect detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Enhanced data transforms for multiple datasets
        self.train_transform = transforms.Compose([
                # 1. All PIL-based ops go first:
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3),
                # 2. Convert PIL Image → Tensor
                transforms.ToTensor(),
                # 3. All tensor-based ops go *after* ToTensor:
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model, datasets, and dataloaders
        self._setup_model()
        self._setup_data()
        self._setup_training()
    
    def _setup_model(self):
        """Setup model and Grad-CAM"""
        self.model = EfficientNetDefectClassifier(
            num_classes=self.config['num_classes'],
            pretrained=True,
            dropout_rate=self.config.get('dropout_rate', 0.3)
        ).to(self.device)
        
        # Setup Grad-CAM
        target_layer = self.model.backbone.features[-1]  # Last conv layer
        self.grad_cam = GradCAM(self.model, target_layer)

    def _get_balanced_class_indices(self, dataset, split='train'):
        """Balance samples across classes for fair training"""
        import random
        from collections import defaultdict

        # Group indices by class
        class_to_indices = defaultdict(list)
        for i, (_, label, _, class_name) in enumerate(dataset.samples):
            class_to_indices[label].append(i)

        # Find minimum class size (or a custom config value)
        min_class_size = min(len(idxs) for idxs in class_to_indices.values())
        target_size = self.config.get('samples_per_class', min_class_size)

        print(f"\nBalancing classes for {split} split:")
        print(f"Target samples per class: {target_size}")

        balanced_indices = []
        for class_id, indices in class_to_indices.items():
            if len(indices) >= target_size:
                sampled = random.sample(indices, target_size)
            else:
                sampled = indices  # Undersampled
                print(f"  Warning: Class {class_id} has only {len(indices)} samples")

            balanced_indices.extend(sampled)
            print(f"  Class {class_id}: {len(sampled)} samples")

        random.shuffle(balanced_indices)
        return balanced_indices

    
    def _setup_data(self):
        """Setup unified datasets and dataloaders with balanced sampling"""
        # Create full datasets first
        full_train_dataset = UnifiedDefectDataset(
            root_dirs=self.config['data_paths'],
            split='train',
            transform=self.train_transform,
            include_datasets=self.config.get('include_datasets', None)
        )
        
        full_test_dataset = UnifiedDefectDataset(
            root_dirs=self.config['data_paths'],
            split='test',
            transform=self.val_transform,
            include_datasets=self.config.get('include_datasets', None)
        )
        
        # Get balanced indices for each split
        train_indices = self._get_balanced_class_indices(full_train_dataset, 'train')
        test_indices = self._get_balanced_class_indices(full_test_dataset, 'test')
        
        # Create subset datasets with balanced sampling
        from torch.utils.data import Subset
        
        self.train_dataset = Subset(full_train_dataset, train_indices)
        self.test_dataset = Subset(full_test_dataset, test_indices)
        
        # Store full datasets for reference
        self.full_train_dataset = full_train_dataset
        self.full_test_dataset = full_test_dataset
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Print dataset statistics
        train_stats = self.full_train_dataset.get_dataset_statistics()
        test_stats = self.full_test_dataset.get_dataset_statistics()
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Original train samples: {train_stats['total_samples']}")
        print(f"Balanced train samples: {len(self.train_dataset)}")
        print(f"Original test samples: {test_stats['total_samples']}")
        print(f"Balanced test samples: {len(self.test_dataset)}")
        print(f"Total classes: {train_stats['total_classes']}")
        print(f"Classes: {list(self.full_train_dataset.class_to_idx.keys())}")
        
        print("\nOriginal Dataset Distribution (Train):")
        for dataset, count in train_stats['dataset_distribution'].items():
            print(f"  {dataset}: {count} samples")
        
        # Calculate balanced distribution
        balanced_train_dist = self._calculate_balanced_distribution(train_indices)
        balanced_test_dist = self._calculate_balanced_distribution(test_indices)
        
        print("\nBalanced Dataset Distribution (Train):")
        for dataset, count in balanced_train_dist.items():
            print(f"  {dataset}: {count} samples")
        
        print("\nBalanced Dataset Distribution (Test):")
        for dataset, count in balanced_test_dist.items():
            print(f"  {dataset}: {count} samples")
        
        # Store for later use
        self.train_stats = train_stats
        self.test_stats = test_stats
        self.balanced_train_dist = balanced_train_dist
        self.balanced_test_dist = balanced_test_dist
    
    def _calculate_balanced_distribution(self, indices):
        """Calculate dataset distribution for balanced indices"""
        distribution = {}
        for idx in indices:
            # Get dataset name from the sample
            _, _, dataset_name, _ = self.full_train_dataset.samples[idx]
            distribution[dataset_name] = distribution.get(dataset_name, 0) + 1
        return distribution
    
    def get_dataset_info(self):
        """Get comprehensive dataset information"""
        return {
            'train_samples': len(self.train_dataset),
            'test_samples': len(self.test_dataset),
            'total_classes': self.train_stats['total_classes'],
            'classes': list(self.full_train_dataset.class_to_idx.keys()),
            'balanced_train_distribution': self.balanced_train_dist,
            'balanced_test_distribution': self.balanced_test_dist,
            'original_train_distribution': self.train_stats['dataset_distribution'],
            'original_test_distribution': self.test_stats['dataset_distribution']
        }    
    
    def _setup_training(self):
        """Setup optimizer and loss function"""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        dataset_correct = {}
        dataset_total = {}
        
        for batch_idx, (data, target, metadata) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Track per-dataset accuracy
            for i, dataset in enumerate(metadata['dataset']):
                if dataset not in dataset_correct:
                    dataset_correct[dataset] = 0
                    dataset_total[dataset] = 0
                
                dataset_correct[dataset] += pred[i].eq(target[i]).item()
                dataset_total[dataset] += 1
        
        # Calculate per-dataset accuracies
        dataset_accuracies = {}
        for dataset in dataset_correct:
            dataset_accuracies[dataset] = 100. * dataset_correct[dataset] / dataset_total[dataset]
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total,
            'dataset_accuracies': dataset_accuracies
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_datasets = []
        dataset_correct = {}
        dataset_total = {}
        
        with torch.no_grad():
            for data, target, metadata in tqdm(self.test_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_datasets.extend(metadata['dataset'])
                
                # Track per-dataset accuracy
                for i, dataset in enumerate(metadata['dataset']):
                    if dataset not in dataset_correct:
                        dataset_correct[dataset] = 0
                        dataset_total[dataset] = 0
                    
                    dataset_correct[dataset] += pred[i].eq(target[i]).item()
                    dataset_total[dataset] += 1
        
        # Calculate metrics
        accuracy = 100. * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )
        
        # Calculate per-dataset accuracies
        dataset_accuracies = {}
        for dataset in dataset_correct:
            dataset_accuracies[dataset] = 100. * dataset_correct[dataset] / dataset_total[dataset]
        
        return {
            'loss': total_loss / len(self.test_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'targets': all_targets,
            'datasets': all_datasets,
            'dataset_accuracies': dataset_accuracies
        }
    
    def train(self):
        """Main training loop"""
        best_acc = 0
        train_history = {'loss': [], 'accuracy': [], 'dataset_accuracies': {}}
        val_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'dataset_accuracies': {}}
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            train_history['loss'].append(train_metrics['loss'])
            train_history['accuracy'].append(train_metrics['accuracy'])
            
            # Store per-dataset accuracies
            for dataset, acc in train_metrics['dataset_accuracies'].items():
                if dataset not in train_history['dataset_accuracies']:
                    train_history['dataset_accuracies'][dataset] = []
                train_history['dataset_accuracies'][dataset].append(acc)
            
            # Validate
            val_metrics = self.validate()
            val_history['loss'].append(val_metrics['loss'])
            val_history['accuracy'].append(val_metrics['accuracy'])
            val_history['precision'].append(val_metrics['precision'])
            val_history['recall'].append(val_metrics['recall'])
            val_history['f1'].append(val_metrics['f1'])
            
            # Store per-dataset accuracies
            for dataset, acc in val_metrics['dataset_accuracies'].items():
                if dataset not in val_history['dataset_accuracies']:
                    val_history['dataset_accuracies'][dataset] = []
                val_history['dataset_accuracies'][dataset].append(acc)
            
            # Update scheduler
            self.scheduler.step(val_metrics['accuracy'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # Print per-dataset accuracies
            print("\nPer-dataset Validation Accuracies:")
            for dataset, acc in val_metrics['dataset_accuracies'].items():
                print(f"  {dataset}: {acc:.2f}%")
            
            # Save best model
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': best_acc,
                    'class_to_idx': self.full_train_dataset.class_to_idx,
                    'config': self.config,
                    'train_stats': self.train_stats,
                    'test_stats': self.test_stats
                }, 'best_model_unified.pth')
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        return train_history, val_history, val_metrics
    
    def plot_training_history(self, train_history: Dict, val_history: Dict, final_metrics: Dict):
        """Plot comprehensive training history"""
        plt.figure(figsize=(8, 6))
        plt.plot(train_history['loss'], label='Train Loss')
        plt.plot(val_history['loss'], label='Val Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('charts/loss_curve.png', dpi=300)
        plt.close()

        # ACCURACY PLOT
        plt.figure(figsize=(8, 6))
        plt.plot(train_history['accuracy'], label='Train Accuracy')
        plt.plot(val_history['accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('charts/accuracy_curve.png', dpi=300)
        plt.close()

        # PRECISION, RECALL, F1
        plt.figure(figsize=(8, 6))
        plt.plot(val_history['precision'], label='Precision')
        plt.plot(val_history['recall'], label='Recall')
        plt.plot(val_history['f1'], label='F1-Score')
        plt.title('Validation Metrics Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('charts/val_metrics.png', dpi=300)
        plt.close()

        # PER-DATASET ACCURACY
        plt.figure(figsize=(10, 6))
        for dataset, accuracies in val_history['dataset_accuracies'].items():
            plt.plot(accuracies, label=f'{dataset}')
        plt.title('Per-Dataset Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('charts/per_dataset_accuracy.png', dpi=300)
        plt.close()

        # CONFUSION MATRIX
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(final_metrics['targets'][-len(self.test_dataset):], 
                            final_metrics['predictions'][-len(self.test_dataset):])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('charts/confusion_matrix.png', dpi=300)
        plt.close()

        # CLASS DISTRIBUTION BAR CHART
        plt.figure(figsize=(10, 6))
        class_counts = list(self.train_stats['class_distribution'].values())
        class_names = list(self.train_stats['class_distribution'].keys())
        plt.bar(range(len(class_counts)), class_counts)
        plt.title('Training Class Distribution')
        plt.xlabel('Class Index')
        plt.ylabel('Samples')
        plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('charts/class_distribution.png', dpi=300)
        plt.close()

    
    def visualize_gradcam(self, num_samples: int = 12):
        """Visualize Grad-CAM results from different datasets"""
        self.model.eval()
        
        # Get samples from each dataset
        dataset_samples = {}
        for data, target, metadata in self.test_loader:
            for i, dataset in enumerate(metadata['dataset']):
                if dataset not in dataset_samples:
                    dataset_samples[dataset] = []
                if len(dataset_samples[dataset]) < num_samples // len(self.config['data_paths']):
                    dataset_samples[dataset].append((data[i], target[i], metadata['class'][i]))
        
        # Create visualization
        n_datasets = len(dataset_samples)
        samples_per_dataset = min(4, num_samples // n_datasets)
        
        fig, axes = plt.subplots(n_datasets * 2, samples_per_dataset, 
                                figsize=(4*samples_per_dataset, 4*n_datasets))
        
        if n_datasets == 1:
            axes = axes.reshape(2, -1)
        
        row = 0
        for dataset_name, samples in dataset_samples.items():
            for col, (img, label, class_name) in enumerate(samples[:samples_per_dataset]):
                img_tensor = img.unsqueeze(0).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(img_tensor)
                    pred = output.argmax(dim=1).item()
                
                # Generate Grad-CAM
                cam = self.grad_cam.generate_cam(img_tensor, pred)
                
                # Denormalize image for visualization
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                
                # Plot original image
                axes[row, col].imshow(img_np)
                axes[row, col].set_title(f'{dataset_name}\nTrue: {class_name}\nPred: {self.full_train_dataset.idx_to_class[pred]}')
                axes[row, col].axis('off')
                
                # Plot Grad-CAM overlay
                axes[row + 1, col].imshow(img_np)
                axes[row + 1, col].imshow(cam, cmap='jet', alpha=0.5)
                axes[row + 1, col].set_title('Grad-CAM Overlay')
                axes[row + 1, col].axis('off')
            
            row += 2
        
        plt.tight_layout()
        plt.savefig('gradcam_visualization_unified.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run unified training"""
    
    # Enhanced configuration for unified training
    config = {
        # Dataset paths - Update these paths according to your dataset structure
        'data_paths': {
            'DAGM': 'Dataset/DAGM',      # Path to DAGM dataset
            'MT': 'Dataset/Magnetic-Tile-Defect',          # Path to Magnetic Tile dataset
            'PCB': 'Dataset/PCB'         # Path to PCB dataset
        },
        
        # Optional: specify which datasets to include (None for all)
        'include_datasets': ['DAGM', 'MT'],  # ['DAGM', 'MT', 'PCB'] or None for all
        
        # Model configuration
        'num_classes': 22,  # Total: DAGM(10) + MT(6) + PCB(6) = 22 classes
        'dropout_rate': 0.3,
        
        # Training configuration
        'batch_size': 4,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 4,
        
        # Data augmentation
        'use_advanced_augmentation': True,
        
        # Model saving
        'save_best_only': True,
        'save_interval': 5  # Save model every N epochs
    }
    
    print("Starting Unified Defect Detection Training...")
    print("="*60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = UnifiedDefectDetectionTrainer(config)
    
    # Train the model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    train_history, val_history, final_metrics = trainer.train()
    
    # Plot training history
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    trainer.plot_training_history(train_history, val_history, final_metrics)
    
    # Visualize Grad-CAM
    trainer.visualize_gradcam(num_samples=12)
    
    # Print comprehensive final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Overall Validation Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Overall Validation Precision: {final_metrics['precision']:.4f}")
    print(f"Overall Validation Recall: {final_metrics['recall']:.4f}")
    print(f"Overall Validation F1-Score: {final_metrics['f1']:.4f}")
    
    print("\nPer-Dataset Performance:")
    for dataset, acc in final_metrics['dataset_accuracies'].items():
        print(f"  {dataset}: {acc:.2f}%")
    
    # Class-wise performance analysis
    print("\nClass-wise Performance Analysis:")
    from sklearn.metrics import classification_report
    
    class_names = [trainer.full_train_dataset.idx_to_class[i] for i in range(len(trainer.full_train_dataset.idx_to_class))]

    report = classification_report(
        final_metrics['targets'], 
        final_metrics['predictions'], 
        target_names=class_names,
        output_dict=True
    )
    
    # Print top and bottom performing classes
    class_f1_scores = {class_name: report[class_name]['f1-score'] 
                       for class_name in class_names if class_name in report}
    
    sorted_classes = sorted(class_f1_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 performing classes:")
    for class_name, f1_score in sorted_classes[:5]:
        print(f"  {class_name}: F1={f1_score:.4f}")
    
    print("\nBottom 5 performing classes:")
    for class_name, f1_score in sorted_classes[-5:]:
        print(f"  {class_name}: F1={f1_score:.4f}")
    
    # Save comprehensive results
    results = {
        'config': config,
        'final_metrics': {
            'overall_accuracy': final_metrics['accuracy'],
            'overall_precision': final_metrics['precision'],
            'overall_recall': final_metrics['recall'],
            'overall_f1': final_metrics['f1'],
            'dataset_accuracies': final_metrics['dataset_accuracies']
        },
        'train_history': train_history,
        'val_history': val_history,
        'train_stats': trainer.train_stats,
        'test_stats': trainer.test_stats,
        'class_mapping': trainer.full_train_dataset.class_to_idx,
        'classification_report': report
    }
    
    with open('results_unified.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results_unified.json")
    print(f"Best model saved to best_model_unified.pth")
    print(f"Training visualizations saved to training_history_unified.png")
    print(f"Grad-CAM visualizations saved to gradcam_visualization_unified.png")
    
    # Additional analysis and recommendations
    print("\n" + "="*60)
    print("ANALYSIS AND RECOMMENDATIONS")
    print("="*60)
    
    # Dataset balance analysis
    train_distribution = trainer.train_stats['dataset_distribution']
    total_train = sum(train_distribution.values())
    
    print("Dataset Balance Analysis:")
    for dataset, count in train_distribution.items():
        percentage = (count / total_train) * 100
        print(f"  {dataset}: {count} samples ({percentage:.1f}%)")
    
    # Recommendations based on performance
    print("\nRecommendations:")
    
    # Check for dataset imbalance
    max_samples = max(train_distribution.values())
    min_samples = min(train_distribution.values())
    if max_samples / min_samples > 2:
        print("  - Consider dataset balancing techniques (oversampling/undersampling)")
    
    # Check for low-performing datasets
    low_performing = [dataset for dataset, acc in final_metrics['dataset_accuracies'].items() 
                     if acc < final_metrics['accuracy'] - 10]
    if low_performing:
        print(f"  - Focus on improving performance for: {', '.join(low_performing)}")
        print("  - Consider dataset-specific augmentation strategies")
    
    # Check overall performance
    if final_metrics['accuracy'] > 90:
        print("  - Excellent performance! Consider deployment or fine-tuning")
    elif final_metrics['accuracy'] > 80:
        print("  - Good performance. Consider additional training epochs or hyperparameter tuning")
    else:
        print("  - Performance needs improvement. Consider:")
        print("    * Increasing training epochs")
        print("    * Adjusting learning rate")
        print("    * Adding more data augmentation")
        print("    * Using a larger model")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

# Additional utility functions for post-training analysis
def load_trained_model(model_path: str, device: torch.device = None):
    """Load a trained unified model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model
    model = EfficientNetDefectClassifier(
        num_classes=len(checkpoint['class_to_idx']),
        pretrained=False,
        dropout_rate=checkpoint['config'].get('dropout_rate', 0.3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['class_to_idx'], checkpoint['config']

def predict_image(model, image_path: str, class_to_idx: Dict, transform, device: torch.device):
    """Predict defect class for a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    # Get class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_class = idx_to_class[predicted_class_idx]
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def analyze_model_performance(results_path: str):
    """Analyze model performance from saved results"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Overall metrics
    final_metrics = results['final_metrics']
    print(f"Overall Accuracy: {final_metrics['overall_accuracy']:.2f}%")
    print(f"Overall F1-Score: {final_metrics['overall_f1']:.4f}")
    
    # Dataset-specific performance
    print("\nDataset-specific Performance:")
    for dataset, acc in final_metrics['dataset_accuracies'].items():
        print(f"  {dataset}: {acc:.2f}%")
    
    # Training convergence analysis
    val_history = results['val_history']
    best_epoch = np.argmax(val_history['accuracy'])
    print(f"\nBest epoch: {best_epoch + 1}")
    print(f"Best validation accuracy: {val_history['accuracy'][best_epoch]:.2f}%")
    
    # Check for overfitting
    train_history = results['train_history']
    final_train_acc = train_history['accuracy'][-1]
    final_val_acc = val_history['accuracy'][-1]
    
    if final_train_acc - final_val_acc > 10:
        print("⚠️  Potential overfitting detected!")
        print(f"   Train accuracy: {final_train_acc:.2f}%")
        print(f"   Validation accuracy: {final_val_acc:.2f}%")
    else:
        print("✅ No significant overfitting detected")
    
    return results

if __name__ == "__main__":
    main()