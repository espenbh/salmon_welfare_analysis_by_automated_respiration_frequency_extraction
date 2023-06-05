# Imports
import sys  
sys.path.insert(0, 'C:\\Users\\espebh\\Documents\\Thesis\\code_ver2')
import cv2
from helpers import m
import colorsys

# Draw bounding boxes and keypoints on an image
# Input is the labels after label augmentation
def draw_label_on_alb_output(augmented):
    # Create color palette
    # Copied from https://stackoverflow.com/questions/876853/generating-color-ranges-in-python
    HSV = [(x/m.NUM_KEYPOINTS, 1, 1) for x in range(m.NUM_KEYPOINTS)]

    img = augmented['image']
    for i in range(len(augmented['bboxes'])):
        cv2.rectangle(img, 
                    (int(augmented['bboxes'][i][0]), int(augmented['bboxes'][i][1])),
                    (int(augmented['bboxes'][i][2]), int(augmented['bboxes'][i][3])), 
                    (255,0,0), 2)
        cv2.putText(img, str(augmented['bbox_id'][i]), (int(augmented['bboxes'][i][0]), int(augmented['bboxes'][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    for i in range(len(augmented['keypoints'])):
        r, g, b = colorsys.hsv_to_rgb(*HSV[int(m.LABEL_MAP[augmented['keypoint_labels'][i]])])
        cv2.circle(img, (int(augmented['keypoints'][i][0]), int(augmented['keypoints'][i][1])), 5, (255*r, 255*g, 255*b), 2)
    return img

# Draw bounding boxes and keypoints on an image
# Input is the labels after incomplete salmons are removed: {keypoints: [[[]]], bboxes: [[]]}
def draw_label_on_keyrcnn_in(labels, img):
    # Create color palette
    # Copied from https://stackoverflow.com/questions/876853/generating-color-ranges-in-python
    HSV = [(x/m.NUM_KEYPOINTS, 1, 1) for x in range(m.NUM_KEYPOINTS)]

    for i in range(len(labels['bboxes'])):
        cv2.rectangle(img, 
                    (int(labels['bboxes'][i][0]), int(labels['bboxes'][i][1])),
                    (int(labels['bboxes'][i][2]), int(labels['bboxes'][i][3])), 
                    (255,0,0), 2)
        cv2.putText(img, str(labels['classes'][i]), (int(labels['bboxes'][i][0]), int(labels['bboxes'][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        for j in range(m.NUM_KEYPOINTS):
            r, g, b = colorsys.hsv_to_rgb(*HSV[j])
            cv2.circle(img, (int(labels['keypoints'][i][j][0]), int(labels['keypoints'][i][j][1])), 5, (255*r, 255*g, 255*b), 2)
    return img

# Draw a cross on an image with center at a given point.
def draw_cross(img, center, line_length = 30, line_color = (0,0,0), line_thickness = 2, line_type = cv2.FONT_HERSHEY_SIMPLEX):
    cv2.line(img, (center[0], center[1]-int(line_length/2)) , (center[0], center[1]+int(line_length/2)), line_color, line_thickness, line_type)
    cv2.line(img, (center[0]-int(line_length/2), center[1]) , (center[0]+int(line_length/2), center[1]) ,  line_color, line_thickness, line_type)
    return img