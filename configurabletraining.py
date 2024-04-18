from ultralytics import YOLO
import torch

# Load a model
model = YOLO("yolov8n.pt")  # nano version of YOLOv8

# Define training parameters
batch_size = 16
learning_rate = 0.001
weight_decay = 0.0005
epochs = 200
num_workers = 4

# Setting up optimizer with weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler setup
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Configure and train the model
model.train(
    data=r"C:\Users\visha\Downloads\Schnecken.v8i.yolov8\data.yaml",
    epochs=epochs,
    batch_size=batch_size,
    optimizer=optimizer,
    scheduler=scheduler,
    num_workers=num_workers
)

# Evaluate model performance on the validation set
metrics = model.val()

# Predict on an image
results = model(r"C:\Users\visha\OneDrive\Desktop\objectdetector\slugtest.png")

# Export the model in ONNX format
path = model.export(format="onnx")

# Print path where the model is saved
print("Model has been exported to:", path)

# Print metrics to understand model performance
print("Validation metrics:", metrics)
