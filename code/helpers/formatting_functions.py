import sys  
sys.path.insert(0, 'C:\\Users\\espebh\\Documents\\Thesis\\code_ver2')
from helpers import m
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision


# Transform the format of the manual labelling into a format that can be passed to the augmentor, and is more practical.
# Input: {shapes: [{label: class, points: [[xmin, ymin], [xmin, xmax]], group_id: c}, {...}]}
# Output: {keypoints: [[x, y], ...], bboxes: [[xmin, ymin, xmax, ymax], ...], keypoint_id: [], keypoint_labels: [], bbox_id: [], class: int}
def labelme_to_alb_in(label, body_is_annotated = False):
    labels_grouped = {}
    labels_grouped['keypoints'] = []
    labels_grouped['bboxes'] = []
    labels_grouped['keypoint_labels'] = []
    labels_grouped['keypoint_id'] = []
    labels_grouped['bbox_id'] = []
    if body_is_annotated:
        labels_grouped['body_bboxes'] = []
        labels_grouped['body_bbox_id'] = []
    for item in label['shapes']:
        if item['label'] in m.LABEL_NAMES: 
            labels_grouped['keypoints'].append(item['points'][0])
            labels_grouped['keypoint_labels'].append(item['label'])
            labels_grouped['keypoint_id'].append(item['group_id'])
        if item['label'] == 'head': 
            single_bbox = list(np.concatenate([item['points'][0], item['points'][1]], axis=0))
            # Ensure all bounding boxes are contained in the frame
            if single_bbox[0] > m.IMG_SHAPE[1]:
                single_bbox[0] = m.IMG_SHAPE[1]
            if single_bbox[2] > m.IMG_SHAPE[1]:
                single_bbox[2] = m.IMG_SHAPE[1]
            if single_bbox[1] > m.IMG_SHAPE[0]:
                single_bbox[1] = m.IMG_SHAPE[0]
            if single_bbox[3] > m.IMG_SHAPE[0]:
                single_bbox[3] = m.IMG_SHAPE[0]
            for number in range(len(single_bbox)):
                if single_bbox[number] < 0:
                    single_bbox[number] = 0
            labels_grouped['bboxes'].append(single_bbox)
            labels_grouped['bbox_id'].append(item['group_id'])
        if item['label'] == 'body' and body_is_annotated: 
            single_body_bbox = list(np.concatenate([item['points'][0], item['points'][1]], axis=0))
            # Ensure all bounding boxes are contained in the frame
            if single_body_bbox[0] > m.IMG_SHAPE[1]:
                single_body_bbox[0] = m.IMG_SHAPE[1]
            if single_body_bbox[2] > m.IMG_SHAPE[1]:
                single_body_bbox[2] = m.IMG_SHAPE[1]
            if single_body_bbox[1] > m.IMG_SHAPE[0]:
                single_body_bbox[1] = m.IMG_SHAPE[0]
            if single_body_bbox[3] > m.IMG_SHAPE[0]:
                single_body_bbox[3] = m.IMG_SHAPE[0]
            for number in range(len(single_body_bbox)):
                if single_body_bbox[number] < 0:
                    single_body_bbox[number] = 0
            labels_grouped['body_bboxes'].append(single_body_bbox)
            labels_grouped['body_bbox_id'].append(item['group_id'])

    # Ensure the bbox points are deterministic
    labels_grouped['bboxes'] = order_bbox_coords(labels_grouped['bboxes'])
    if body_is_annotated:
        labels_grouped['body_bboxes'] = order_bbox_coords(labels_grouped['body_bboxes'])

    # If multiple occluded fish, we remove the entire image for simplicity
    if len([x for x in labels_grouped['keypoint_id'] if x == 14])>7:
        labels_grouped = None
    return labels_grouped

# Ensure the top left and bottom right points in the bounding box is the defining points. x_min<x_max and y_min<y_max.
def order_bbox_coords(bboxes):
    bboxes_ordered = []
    for bbox in bboxes:
        if bbox[0] > bbox[2]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        bboxes_ordered.append(bbox)
    return bboxes_ordered


# Function to remove incomplete labels after augmentations
# It also structures the labels for network training
# Input is the output of the augmentor: {image: array, bboxes: [(xmin, ymin, xmax, ymax), ...], bbox_labels: [], bbox_id: [], keypoints: [(x, y), ...], keypoint_id: [], keypoint_labels: []}
# Output is the full classes, keypoints and bounding boxes: {keypoints: [[[]]], bboxes: [[]], classes: [], areas = }. 
# Keypoints are ordered as [boxidx][keyidx], with keyidx: ljaw: 0, ujaw: 1, eye: 2.
# bboxes are numbered after the salmon group idx
def alb_out_to_keyrcnn_in(label):
    # Calculate area of the bounding boxes
    # Remove bounding boxes if it has no area
    bbox_dict = {}
    for i in range(len(label['bboxes'])):
        curr_area = (label['bboxes'][i][2] - label['bboxes'][i][0])*(label['bboxes'][i][3] - label['bboxes'][i][1])
        if curr_area > 0.01:
            bbox_dict[str(label['bbox_id'][i])] = list(label['bboxes'][i])

    # Initialize dictionaries
    keypoint_labels = [m.LABEL_MAP[x] for x in label['keypoint_labels']] # Assign a numerical value, describing type, to each keypoint [2, 0, 1, 3, 2, ...]
    keypoint_labels_dict = dict([(y,[]) for y in bbox_dict.keys()])    # Initialize a dictionary to store keypoint labels assigned class-wise {1: [], 2: [], ...}
    keypoints_dict = dict([(y,[]) for y in bbox_dict.keys()])          # Initialize a dictionary to store keypoints assigned class-wise {1: [], 2: [], ...}

    # Fill in the dictionaries
    for idx in range(len(label['keypoint_id'])): 
        if str(label['keypoint_id'][idx]) in keypoint_labels_dict:
            keypoint_labels_dict[str(label['keypoint_id'][idx])].append(keypoint_labels[idx])
            keypoints_dict[str(label['keypoint_id'][idx])].append(np.concatenate([label['keypoints'][idx], [1]]).tolist())

    # Remove salmon if not complete
    keypoint_labels_dict = dict([(y,x) for y,x in keypoint_labels_dict.items() if len(x) == m.NUM_KEYPOINTS]) 
    keypoints_dict = dict([(y,x) for y,x in keypoints_dict.items() if len(x) == m.NUM_KEYPOINTS])

    # Create a dictionary containing only the bounding boxes and areas with a full set of keypoints
    bbox_dict = dict([(y,x) for y,x in bbox_dict.items() if y in keypoints_dict])

    # Turn the dictionaries into lists
    bboxes = list(bbox_dict.values())
    classes = list(bbox_dict.keys())
    keypoint_labels_list = list(keypoint_labels_dict.values())
    keypoints_list = list(keypoints_dict.values())
    ordered_keypoints = [[[] for i in range(len(keypoints_list[0]))] for j in range(len(keypoints_list))]

    # Find the areas of the valid bboxes
    area = [(label['bboxes'][i][2] - label['bboxes'][i][0])*(label['bboxes'][i][3] - label['bboxes'][i][1]) for i in range(len(bboxes))]
    
    # Order keypoints according to keypoint labels
    for boxidx in range(len(keypoint_labels_list)):
        for keyidx, i in enumerate(keypoint_labels_list[boxidx]):
            ordered_keypoints[boxidx][i] = keypoints_list[boxidx][keyidx]
    
    new_labels = {'bboxes': bboxes, 'keypoints': ordered_keypoints, 'classes': classes, 'areas': area}
    return new_labels

# Transform image to a valid neural network input for keypoint rcnn
def cv2_to_keyrcnn_in(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_tensor = torchvision.transforms.functional.pil_to_tensor(img_pil)
    img_tensor = torch.div(img_tensor, 255)
    return img_tensor

# Transform image to a valid neural network input for the network performing classification from dot masks
def cv2_to_dot_network_in(images):
    net_in_imgs = []
    for image in images:
        img_gray = cv2.normalize(cv2.cvtColor(image.astype(np.float32).copy(), cv2.COLOR_BGR2GRAY).astype(float), None, 0, 1, cv2.NORM_MINMAX)
        img_mask = generate_dot_masks_from_gray_images([img_gray])[0]
        img_laplacian = cv2.normalize(cv2.Laplacian(img_gray.copy(),cv2.CV_64F).astype(float), None, 0, 1, cv2.NORM_MINMAX)
        img_net_in = np.dstack([img_gray, img_mask, img_laplacian])
        net_in_imgs.append(img_net_in)
    return net_in_imgs

# Transform image to a valid neural network input for the network performing classification from RGB images
def cv2_to_RGB_network_in(images):
    net_in_imgs = []
    for image in images:
        img_pil = Image.fromarray(image.astype(np.uint8))
        img_tensor = torchvision.transforms.functional.pil_to_tensor(img_pil)
        img_tensor = torch.div(img_tensor, 255)
        img_tensor = img_tensor.to(m.DEVICE)
        net_in_imgs.append(img_tensor)
    return net_in_imgs

# Performs adaptive thresholding on a gray scale image
def generate_dot_masks_from_gray_images(images):
    dot_masks = []
    for image in images:
        thresh = cv2.adaptiveThreshold(np.multiply(image, 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,21)
        threshold_img = np.array(image<thresh).astype(np.uint8)
        dot_masks.append(threshold_img)
    return dot_masks