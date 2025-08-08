import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.ops import box_iou
import time

class SaigaDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        
        # Include both jpg and jpeg extensions
        self.images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            self.images.extend(list(self.data_dir.glob(f'**/{ext}')))
        
        print(f"Found {len(self.images)} images")
        
        # Check for corresponding XML files
        valid_images = []
        self.annotations = {}
        
        for img_path in self.images:
            xml_path = img_path.parent / f"{img_path.stem}.xml"
            if xml_path.exists():
                valid_images.append(img_path)
                self.annotations[img_path.stem] = xml_path
            else:
                print(f"Warning: No annotation file found for {img_path.name}")
        
        self.images = valid_images
        print(f"Found {len(self.images)} images with valid annotations")
        
    def __len__(self):
        return len(self.images)
    
    def parse_xml(self, xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            boxes = []
            labels = []
            
            # Get image dimensions for validation
            size_elem = root.find('size')
            if size_elem is not None:
                img_width = int(size_elem.find('width').text)
                img_height = int(size_elem.find('height').text)
            else:
                img_width = img_height = None
            
            for obj in root.findall('object'):
                label = obj.find('name').text.lower().strip()  # Normalize label
                bbox = obj.find('bndbox')
                
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Validate bounding box
                if xmin >= xmax or ymin >= ymax:
                    print(f"Invalid bounding box in {xml_file}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                    continue
                
                if img_width and img_height:
                    if xmax > img_width or ymax > img_height or xmin < 0 or ymin < 0:
                        print(f"Bounding box outside image bounds in {xml_file}")
                        continue
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
            
            if not boxes:
                print(f"No valid boxes found in {xml_file}")
                return torch.zeros((0, 4), dtype=torch.float32), []
            
            return torch.tensor(boxes, dtype=torch.float32), labels
            
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return torch.zeros((0, 4), dtype=torch.float32), []
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            original_size = image.size  # (width, height)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        
        # Get corresponding XML file
        xml_path = self.annotations[img_path.stem]
        boxes, labels = self.parse_xml(xml_path)
        
        # Skip images with no valid annotations
        if len(boxes) == 0:
            print(f"Skipping {img_path.name} - no valid annotations")
            # Return next item instead
            return self.__getitem__((idx + 1) % len(self.images))
        
        # Convert labels to tensor
        label_map = {'saiga': 1, 'background': 0}  # 1-based indexing for object classes
        
        try:
            label_ids = torch.tensor([label_map.get(label, 1) for label in labels], dtype=torch.int64)
        except Exception as e:
            print(f"Error processing labels {labels}: {e}")
            label_ids = torch.ones(len(labels), dtype=torch.int64)  # Default to class 1
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            # Scale boxes to match resized image
            if hasattr(self.transform.transforms[0], 'size'):
                new_size = self.transform.transforms[0].size
                if isinstance(new_size, int):
                    new_size = (new_size, new_size)
                
                scale_x = new_size[0] / original_size[0]
                scale_y = new_size[1] / original_size[1]
                
                boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
                boxes[:, [1, 3]] *= scale_y  # Scale y coordinates
        
        # Ensure boxes are valid
        boxes[:, 2:] = torch.max(boxes[:, 2:], boxes[:, :2] + 1)  # Ensure xmax > xmin, ymax > ymin
        
        target = {
            'boxes': boxes,
            'labels': label_ids,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        return image, target

def get_model(num_classes=2):
    """Create and return the model"""
    # Load pre-trained model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    """Custom collate function for object detection"""
    return tuple(zip(*batch))

def validate_model(model, val_loader, device):
    """Validate the model and return average loss"""
    model.train()  # Keep in train mode to get losses
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def train_model():
    print("Initializing training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Improved data transforms with better augmentation
    train_transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = SaigaDataset("images/train", transform=None)
    
    if len(full_dataset) == 0:
        print("ERROR: No valid training data found!")
        return None
    
    print(f"Total valid images found: {len(full_dataset)}")
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Create indices for splitting
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets with appropriate transforms
    train_dataset = torch.utils.data.Subset(
        SaigaDataset("images/train", transform=train_transform), 
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        SaigaDataset("images/train", transform=val_transform), 
        val_indices
    )
    
    print(f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}")
    
    # Create data loaders with better settings
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,  # Increased batch size
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("Initializing model...")
    model = get_model(num_classes=2)  # background + saiga
    model = model.to(device)
    
    # Better optimizer settings
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    print("Model initialized successfully")
    
    # Training loop with validation
    print("Starting training loop...")
    num_epochs = 15  # Increased epochs
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        start_time = time.time()
        
        try:
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Skip empty batches
                if not images or not targets:
                    continue
                
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"Warning: Invalid loss at batch {batch_idx}")
                    continue
                
                losses.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += losses.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {losses.item():.4f}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            continue
        
        epoch_time = time.time() - start_time
        
        if batch_count > 0:
            avg_train_loss = running_loss / batch_count
            train_losses.append(avg_train_loss)
            
            # Validation phase
            print("Running validation...")
            val_loss = validate_model(model, val_loader, device)
            val_losses.append(val_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'saiga_detection_best.pth')
                print("Saved best model")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
            
            # Step learning rate scheduler
            lr_scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), 'saiga_detection_final.pth')
    print("\nTraining completed!")
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_curves.png')
    plt.show()
    
    return model

def test_model(model_path='saiga_detection_best.pth', test_dir='images/train', confidence_threshold=0.3):
    """Test the trained model on test images with lower threshold"""
    print(f"\nTesting model with confidence threshold: {confidence_threshold}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = get_model(num_classes=2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found!")
        return
    
    model = model.to(device)
    model.eval()
    
    # Transform for test images
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get test images
    test_path = Path(test_dir)
    test_images = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        test_images.extend(list(test_path.glob(f'**/{ext}')))
    
    if not test_images:
        print(f"No test images found in {test_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Create output directory
    output_dir = Path('images/test_results')
    output_dir.mkdir(exist_ok=True)
    
    class_names = ['background', 'saiga']
    total_detections = 0
    
    with torch.no_grad():
        for i, img_path in enumerate(test_images[:10]):  # Test first 10 images
            print(f"\nProcessing {img_path.name} ({i+1}/10)")
            
            # Load and preprocess image
            try:
                original_image = Image.open(img_path).convert('RGB')
                image = transform(original_image).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
            
            # Make prediction
            predictions = model(image)
            
            # Process predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            print(f"Raw predictions: {len(boxes)} boxes")
            print(f"Score range: {scores.min():.3f} - {scores.max():.3f}")
            
            # Filter predictions by confidence threshold
            keep_indices = scores >= confidence_threshold
            filtered_boxes = boxes[keep_indices]
            filtered_scores = scores[keep_indices]
            filtered_labels = labels[keep_indices]
            
            total_detections += len(filtered_boxes)
            print(f"Filtered detections: {len(filtered_boxes)} (threshold: {confidence_threshold})")
            
            # Visualize results
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(original_image)
            
            # Show all detections above threshold
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                # Scale boxes back to original image size
                orig_w, orig_h = original_image.size
                x1, y1, x2, y2 = box
                x1 = x1 * orig_w / 800
                y1 = y1 * orig_h / 800
                x2 = x2 * orig_w / 800
                y2 = y2 * orig_h / 800
                
                # Draw bounding box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                ax.text(x1, y1-10, f'{class_names[label]}: {score:.2f}', 
                       color='red', fontsize=12, weight='bold',
                       bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_title(f'Detections: {img_path.name} (Found: {len(filtered_boxes)})')
            ax.axis('off')
            
            # Save result
            output_path = output_dir / f'result_{img_path.stem}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
    
    print(f"\nTotal detections across all test images: {total_detections}")
    print(f"Average detections per image: {total_detections/min(10, len(test_images)):.1f}")

def test_single_image(image_path, model_path='saiga_detection_best.pth', confidence_threshold=0.3, save_result=True):
    """
    Test a single image with the trained Saiga detection model
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the trained model weights
        confidence_threshold (float): Minimum confidence score for detections
        save_result (bool): Whether to save the result image
    
    Returns:
        dict: Detection results with boxes, scores, and labels
    """
    print(f"Testing image: {image_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return None
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare model
    print("Loading model...")
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Image preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    print("Loading and preprocessing image...")
    try:
        original_image = Image.open(image_path).convert('RGB')
        original_size = original_image.size  # (width, height)
        print(f"Original image size: {original_size}")
        
        # Apply transforms
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Make prediction
    print("Running inference...")
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process predictions
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    
    print(f"Raw predictions: {len(boxes)} boxes")
    if len(boxes) > 0:
        print(f"Score range: {scores.min():.3f} - {scores.max():.3f}")
        
        # Show top 5 raw predictions
        top_indices = np.argsort(scores)[::-1][:5]
        print("Top 5 raw predictions:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Score: {scores[idx]:.3f}, Label: {labels[idx]}")
    
    # Filter predictions by confidence threshold
    keep_indices = scores >= confidence_threshold
    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]
    filtered_labels = labels[keep_indices]
    
    print(f"Filtered detections: {len(filtered_boxes)} (threshold: {confidence_threshold})")
    
    # Class names
    class_names = ['background', 'saiga']
    
    # Visualize results
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(original_image)
    
    detection_results = []
    
    # Draw bounding boxes for filtered detections
    for i, (box, score, label) in enumerate(zip(filtered_boxes, filtered_scores, filtered_labels)):
        # Scale boxes back to original image size
        orig_w, orig_h = original_size
        x1, y1, x2, y2 = box
        x1 = x1 * orig_w / 800
        y1 = y1 * orig_h / 800
        x2 = x2 * orig_w / 800
        y2 = y2 * orig_h / 800
        
        # Store detection info
        detection_info = {
            'box': [x1, y1, x2, y2],
            'score': float(score),
            'label': int(label),
            'class_name': class_names[label]
        }
        detection_results.append(detection_info)
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label with confidence score
        label_text = f'{class_names[label]}: {score:.2f}'
        ax.text(x1, y1-10, label_text, 
               color='red', fontsize=14, weight='bold',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', linewidth=1))
        
        print(f"Detection {i+1}: {class_names[label]} (confidence: {score:.3f})")
        print(f"  Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Set title and styling
    image_name = Path(image_path).name
    ax.set_title(f'Saiga Detection Results: {image_name}\n'
                f'Found: {len(filtered_boxes)} detection(s)', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save result if requested
    if save_result:
        output_dir = Path('single_image_results')
        output_dir.mkdir(exist_ok=True)
        
        output_name = f'result_{Path(image_path).stem}.png'
        output_path = output_dir / output_name
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        print(f"Result saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Return results
    results = {
        'image_path': image_path,
        'detections': detection_results,
        'total_detections': len(detection_results),
        'confidence_threshold': confidence_threshold
    }
    
    return results

if __name__ == "__main__":
    try:
        # print("="*50)
        # print("IMPROVED SAIGA DETECTION TRAINING")
        # print("="*50)
        
        # # Train the model
        # model = train_model()
        
        # if model is not None:
        #     # Test with lower confidence threshold
        #     print("\n" + "="*50)
        #     print("TESTING WITH LOWER THRESHOLD")
        #     print("="*50)
        #     test_model(confidence_threshold=0.3)  # Lower threshold
            
        # print("\n" + "="*50)
        # print("TRAINING AND TESTING COMPLETED!")
        # print("="*50)
        test_single_image(image_path='QCupBlue__21093.jpg')

        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()