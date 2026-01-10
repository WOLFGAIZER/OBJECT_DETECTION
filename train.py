from ultralytics import YOLO

def train_model():
    DATA_YAML_PATH = r"C:\Users\sai78\Desktop\OBJECT_DETECTION\custom.yaml" 
    
    model = YOLO("yolov8n.yaml") 

    model.train(
        data=DATA_YAML_PATH,
        epochs=150,           
        imgsz=416,            
        batch=16,              
        pretrained=False,     
        optimizer='AdamW',   
        project=r"C:\Users\sai78\Desktop\OBJECT_DETECTION\runs", 
        name="FruitDetection_Results",
        exist_ok=True,
        lr0=0.001,            
        device=0,
        scale = 0.5,
        degrees = 15.0,
        mosaic = 1.0,
        mixup = 0.1,
        blur = 0.1             
    )

if __name__ == '__main__':
    train_model()