from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch n means nano version

# Use the model
model.train(data=r"C:\Users\visha\Downloads\Schnecken.v8i.yolov8\data.yaml", epochs=200)  # train the model
metrics =model.val()  # evaluate model performance on the validation set
results =model(r"C:\Users\visha\OneDrive\Desktop\objectdetector\slugtest.png")  # predict on an image
path = model.export(format="onnx")
