from inference_sdk import InferenceHTTPClient
import cv2
import os

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="UngFLJeG40iLjnZpv83b"
)

# Directory path where images are stored
image_dir = r"C:\objectdetector\huntbot.v1i.yolov8\test\images"
#"C:\Users\visha\Downloads\SlugsBot.v2i.yolov8\test\images"
# List all image files in the directory
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_path in image_files:
    # run inference on a local image
    response = CLIENT.infer(
        image_path, 
        model_id="huntbot-sa0cj/2"
    )
    print (response)

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Error loading image: ", image_path)
    else:
        # Iterate over the detections and draw bounding boxes
        for prediction in response.get('predictions', []):
            x_center, y_center = int(prediction['x']), int(prediction['y'])
            width, height = int(prediction['width']), int(prediction['height'])
            
            # Convert center coordinates to top left corner
            x_start = x_center - width // 2
            y_start = y_center - height // 2
            
            # Define the rectangle points
            start_point = (x_start, y_start)
            end_point = (x_start + width, y_start + height)
            
            # Draw the rectangle on the image
            cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)

        # Display the image with bounding boxes
        cv2.imshow("Detections", image)
        cv2.waitKey(1000)  # Display each image for 1000 milliseconds (1 second)
        cv2.destroyAllWindows()
