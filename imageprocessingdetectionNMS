import cv2
import numpy as np

def filter_contours_by_area(contours, min_area):
    """Filter out contours that have a smaller area than the minimum threshold."""
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y)
        # coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

# Load the image from the path
img = cv2.imread(r"C:\Users\visha\Downloads\huntbot.v1i.yolov8\test\images\H168B40WFHVD_jpg.rf.d9c5a0127a8004d886814b9ef2325d33.jpg")

# Resize the image to 640x640 pixels
img_resized = cv2.resize(img, (640, 640))

# Convert the resized image to grayscale
gray_img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection on the grayscale image
edges_resized = cv2.Canny(gray_img_resized, 120, 180)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Apply dilation followed by erosion, also known as closing
morph_closed = cv2.morphologyEx(edges_resized, cv2.MORPH_CLOSE, kernel)

# Find contours from the morphologically processed image
contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on a minimum area
filtered_contours = filter_contours_by_area(contours, min_area=120)

# Draw bounding boxes around the filtered contours
boxes = []
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    boxes.append([x, y, x+w, y+h])

# Apply non-maximum suppression to the bounding boxes
boxes_nms = non_max_suppression_fast(np.array(boxes), overlapThresh=0.1)

# Draw the final bounding boxes
for (startX, startY, endX, endY) in boxes_nms:
    cv2.rectangle(img_resized, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Display the original image with bounding boxes
cv2.imshow('Detected Slugs', img_resized)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
