from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # nano version of YOLOv8

# Define the path to your data and model configuration
data_path = r"C:\Users\visha\Downloads\Schnecken.v8i.yolov8\data.yaml"

# Configure and train the model using only supported arguments
model.train(
    data=data_path,
    model="yolov8n.pt",  # Specify the model if not already loaded
    epochs=200,
    batch=16,  # Use 'batch' for batch size
    imgsz=640,  # Define the image size here directly
    workers=4,  # Define the number of workers
    lr0=0.001  # Initial learning rate
)

# Evaluate model performance on the validation set
metrics = model.val()

# Predict on an image
results = model.predict(r"C:\Users\visha\OneDrive\Desktop\objectdetector\slugtest.png")

# Export the model in ONNX format
path = model.export(format="onnx")
print(f"Model has been exported to: {path}")
print("Validation metrics:", metrics)
