import cv2
import numpy as np
import os  # For directory operations

def filter_contours_by_area(contours, min_area):
    """Filter out contours that have a smaller area than the minimum threshold."""
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

# Define the directory path that contains the images
directory_path = r"C:\Users\visha\Downloads\SlugsBot.v2i.yolov8\train\images"
#"C:\Users\visha\Downloads\SlugsBot.v2i.yolov8\test\images"

# Iterate over each image in the directory 
for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(directory_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img_resized = cv2.resize(img, (640, 640))
            gray_img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            edges_resized = cv2.Canny(gray_img_resized, 80, 100)
            kernel = np.ones((4, 4), np.uint8)
            morph_closed = cv2.morphologyEx(edges_resized, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = filter_contours_by_area(contours, min_area=120)
            boxes = [cv2.boundingRect(cnt) for cnt in filtered_contours]
            boxes_array = np.array(boxes).reshape(-1, 4)
            boxes_nms = non_max_suppression_fast(boxes_array, overlapThresh=0.1)

            for (x, y, w, h) in boxes_nms:
                cv2.rectangle(img_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Detected Slugs', img_resized)
            if cv2.waitKey(1000) == 27:  # wait 3 seconds or until ESC key is pressed
                break

cv2.destroyAllWindows()
