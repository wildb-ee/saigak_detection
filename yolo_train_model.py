import torch
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import yaml
from ultralytics import YOLO
import shutil

class XMLToYOLOConverter:
    def __init__(self, xml_dir, images_dir, output_dir):
        self.xml_dir = Path(xml_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.class_names = {'saiga': 0, 'cat': 1, 'dog': 2, 'person':3, 'camel': 4, 'goose' : 5, 'chicken': 6, 'horse': 7, 'cow': 8, 'sheep': 9}  # Add more classes as needed

    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """Convert XML bbox to YOLO format (normalized center x, center y, width, height)"""
        xmin, ymin, xmax, ymax = bbox
        
        # Calculate center coordinates and dimensions
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # Normalize by image dimensions
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height
        
        return center_x, center_y, width, height
    
    def parse_xml(self, xml_file):
        """Parse XML annotation file"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        annotations = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label not in self.class_names:
                continue
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format
            center_x, center_y, width, height = self.convert_bbox_to_yolo(
                [xmin, ymin, xmax, ymax], img_width, img_height
            )
            
            class_id = self.class_names[label]
            annotations.append([class_id, center_x, center_y, width, height])
            
        return annotations
    
    def convert_dataset(self):
        """Convert entire dataset from XML to YOLO format"""
        # Create output directories
        train_images_dir = self.output_dir / 'images' / 'train'
        train_labels_dir = self.output_dir / 'labels' / 'train'
        val_images_dir = self.output_dir / 'images' / 'val'
        val_labels_dir = self.output_dir / 'labels' / 'val'
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            image_files.extend(list(self.images_dir.glob(f'**/{ext}')))
        
        print(f"Found {len(image_files)} images")
        
        # Split dataset (80% train, 20% val)
        train_size = int(0.8 * len(image_files))
        train_files = image_files[:train_size]
        val_files = image_files[train_size:]
        
        # Process training files
        print("Processing training files...")
        self.process_files(train_files, train_images_dir, train_labels_dir)
        
        # Process validation files
        print("Processing validation files...")
        self.process_files(val_files, val_images_dir, val_labels_dir)
        
        # Create dataset configuration file
        self.create_yaml_config()
        
        print(f"Dataset converted successfully!")
        print(f"Training images: {len(train_files)}")
        print(f"Validation images: {len(val_files)}")
    
    def process_files(self, image_files, images_dir, labels_dir):
        """Process a list of image files and their annotations"""
        for img_path in image_files:
            # Copy image file
            shutil.copy2(img_path, images_dir / img_path.name)
            
            # Find corresponding XML file
            xml_path = img_path.parent / f"{img_path.stem}.xml"
            
            if xml_path.exists():
                # Parse XML and create YOLO annotation
                annotations = self.parse_xml(xml_path)
                
                # Write YOLO format annotation file
                txt_path = labels_dir / f"{img_path.stem}.txt"
                with open(txt_path, 'w') as f:
                    for ann in annotations:
                        f.write(' '.join(map(str, ann)) + '\n')
            else:
                # Create empty annotation file if no XML exists
                txt_path = labels_dir / f"{img_path.stem}.txt"
                txt_path.touch()
    
    def create_yaml_config(self):
        """Create YOLO dataset configuration file"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': list(self.class_names.keys())
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset config saved to: {yaml_path}")

def train_yolo_model():
    """Train YOLO model on saiga dataset"""
    print("Starting YOLO training...")
    
    # Convert XML dataset to YOLO format
    print("Converting dataset to YOLO format...")
    converter = XMLToYOLOConverter(
        xml_dir="images/train",
        images_dir="images/train", 
        output_dir="yolo_dataset"
    )
    converter.convert_dataset()
    
    # Initialize YOLO model
    print("Initializing YOLO model...")
    model = YOLO('yolov8n.pt')  # Use yolov8s.pt, yolov8m.pt, yolov8l.pt, or yolov8x.pt for larger models
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data='yolo_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='saiga_detection',
        name='yolov8_saiga',
        save=True,
        plots=True,
        val=True,
        patience=10,  # Early stopping patience
        save_period=10,  # Save checkpoint every 10 epochs
        workers=8,
        optimizer='AdamW',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=2.0,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        augment=True
    )
    
    print("Training completed!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    
    # Validate the model
    print("Running validation...")
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")

def test_model(model_path, test_image_path):
    """Test the trained model on a single image"""
    model = YOLO(model_path)
    
    # Run inference
    results = model(test_image_path)
    
    # Display results
    for r in results:
        r.show()  # Display image with predictions
        
    return results

if __name__ == "__main__":
    try:

        
        #train_yolo_model()
        
        test_results = test_model('saiga_detection/yolov8_saiga/weights/best.pt', 'test11.webp')
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
