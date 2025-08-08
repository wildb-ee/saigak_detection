import cv2
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        objects.append((label, xmin, ymin, xmax, ymax))
    
    return objects

def visualize_image_with_annotations(image_path, xml_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    annotations = parse_voc_xml(xml_path)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for label, xmin, ymin, xmax, ymax in annotations:
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, label, color='red', fontsize=12, backgroundcolor='white')

    plt.axis('off')
    plt.savefig("images/visualized_image.png", bbox_inches='tight', pad_inches=0.1)

image_filename = 'images/train/1d0e27fcbcbc2af3ad4545b38d7dd058.jpg'
xml_filename = 'images/train/1d0e27fcbcbc2af3ad4545b38d7dd058.xml'
visualize_image_with_annotations(image_filename, xml_filename)
