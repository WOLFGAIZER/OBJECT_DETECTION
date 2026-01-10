import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

#configuration
BASE_DIR = r"C:\Users\sai78\Desktop\OBJECT_DETECTION\datasets"
IMG_DIR = os.path.join(BASE_DIR, "images")
ANN_DIR = os.path.join(BASE_DIR, "annotations") 

CLASSES = ["apple", "banana", "orange"] 
CLASS_MAP = {"banana": 0, "snake fruit": 1, "dragon fruit": 2, "pineapple": 3}

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def process_dataset():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(BASE_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, split, 'labels'), exist_ok=True)

    images = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    def move_and_convert(file_list, split):
        for img_name in file_list:
            name_no_ext = os.path.splitext(img_name)[0]
            xml_file = os.path.join(ANN_DIR, name_no_ext + ".xml")
            
            if not os.path.exists(xml_file): continue

            shutil.copy(os.path.join(IMG_DIR, img_name), os.path.join(BASE_DIR, split, 'images', img_name))

            tree = ET.parse(xml_file) #converting the XML files into TXT for using them
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            with open(os.path.join(BASE_DIR, split, 'labels', name_no_ext + ".txt"), 'w') as f:
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text.lower()
                    if cls_name not in CLASS_MAP: continue
                    cls_id = CLASS_MAP[cls_name]
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert_bbox((w, h), b)
                    f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

    print("Splitting and converting data...")
    move_and_convert(train_imgs, 'train')
    move_and_convert(val_imgs, 'val')
    print("Done! Data is now in datasets/train and datasets/val")

if __name__ == "__main__":
    process_dataset()