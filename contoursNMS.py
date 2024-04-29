import cv2
import numpy as np

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

# Load the image from the path
img = cv2.imread(r"C:\Users\visha\Downloads\SlugsBot.v1i.yolov8\test\images\NMSprediction.jpg")
img_resized = cv2.resize(img, (640, 640))
gray_img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
edges_resized = cv2.Canny(gray_img_resized, 120, 150)
kernel = np.ones((5, 5), np.uint8)
morph_closed = cv2.morphologyEx(edges_resized, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = filter_contours_by_area(contours, min_area=170)

# Draw the filtered contours in a different color
for cnt in filtered_contours:
    cv2.drawContours(img_resized, [cnt], -1, (255, 0, 0), 2)  # Blue color for contours

boxes = []
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    boxes.append([x, y, x+w, y+h])
boxes_nms = non_max_suppression_fast(np.array(boxes), overlapThresh=0.5)

# Draw the final bounding boxes
for (startX, startY, endX, endY) in boxes_nms:
    cv2.rectangle(img_resized, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Green color for bounding boxes

cv2.imshow('Detected Slugs', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
