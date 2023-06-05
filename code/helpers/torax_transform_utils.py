import sys  
sys.path.insert(0, 'C:\\Users\\espebh\\Documents\\Thesis\\code')
import numpy as np
import albumentations as alb
from helpers.formatting_functions import alb_out_to_keyrcnn_in, labelme_to_alb_in
from helpers import m
import os
import json
import cv2
import copy


# Find out which way the salmon is swimming
def swim_dir(label_keyrcnn_in, fish_idx):
    eye_x = label_keyrcnn_in['keypoints'][fish_idx][2][0]
    dfin_x = label_keyrcnn_in['keypoints'][fish_idx][6][0]
    if eye_x > dfin_x:
        swim_dir = 'sr'
    else:
        swim_dir = 'sl'
    return swim_dir


# Define a baseline set of keypoints that all other fish will be mapped onto
def extract_target_transformation(img_name, fish_idx, path):
    # Define the keypoints that should be used for the transformation
    # eye, root of pectoral fin, head-body-interface and dorsal fin

    eye_idx = m.LABEL_NAMES.index('eye')
    rpec_idx = m.LABEL_NAMES.index('rpec')
    hbi_idx = m.LABEL_NAMES.index('head_body_intercept')
    dfin_idx = m.LABEL_NAMES.index('dfin')
    kp_for_hg = [eye_idx, rpec_idx, hbi_idx, dfin_idx]

    # Load image and manual annotations from disk
    img = cv2.imread(os.path.join(path, 'images', img_name + '.jpg'))
    with open(os.path.join(path, 'labels', img_name + '.json'), 'r') as f:
        label = json.load(f)

    # change annotation format
    label_alb_in = labelme_to_alb_in(label, body_is_annotated=False)
    label_keyrcnn_in = alb_out_to_keyrcnn_in(label_alb_in)
    # find the simming direction
    #dir = swim_dir(label_keyrcnn_in, fish_idx)

    # Extract baseline for the original swimming direction
    kp_base_orig = np.array(label_keyrcnn_in['keypoints'])[fish_idx][kp_for_hg,0:2].astype(np.float32)
    bbox_base_orig = label_keyrcnn_in['bboxes'][fish_idx]

    # Flip the image and annotations
    augmentor = alb.Compose([alb.augmentations.geometric.transforms.HorizontalFlip(p=1)],
                            bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['bbox_labels', 'bbox_id']), #[x_min, y_min, x_max, y_max]
                            keypoint_params=alb.KeypointParams(format='xy',  label_fields=['keypoint_labels', 'keypoint_id'])) # [x, y]
    label_alb_out = augmentor(image=img, bboxes=label_alb_in['bboxes'],
                        bbox_labels = label_alb_in['bboxes'], 
                        bbox_id =  label_alb_in['bbox_id'], 
                        keypoints = label_alb_in['keypoints'], 
                        keypoint_labels=label_alb_in['keypoint_labels'], 
                        keypoint_id=label_alb_in['keypoint_id'])

    # Extract baseline for the augmented swimming direction
    label_keyrcnn_in_aug = alb_out_to_keyrcnn_in(label_alb_out)
    kp_base_aug = np.array(label_keyrcnn_in_aug['keypoints'])[fish_idx][kp_for_hg,0:2].astype(np.float32)
    bbox_base_aug = label_keyrcnn_in_aug['bboxes'][fish_idx]

    eye_x = label_keyrcnn_in['keypoints'][fish_idx][2][0]
    dfin_x = label_keyrcnn_in['keypoints'][fish_idx][6][0]
    if eye_x > dfin_x:
        base = {'kp_base_sr': kp_base_orig, 'bbox_base_sr': bbox_base_orig, 'kp_base_sl': kp_base_aug, 'bbox_base_sl': bbox_base_aug}
    else:
        base = {'kp_base_sl': kp_base_orig, 'bbox_base_sl': bbox_base_orig, 'kp_base_sr': kp_base_aug, 'bbox_base_sr': bbox_base_aug}
    return base


# Warp all fish in a tracker onto a fixed set of keypoints (base)
# Returns a list of images, and a list of box numbers. When the list indexes are the same, the fish is the same.
# The keypoints are the eye, the root of the pectoral fin proximal to the eye, the ventral root of the dorsal fin, and the interface between the head and body.
def warp_active_tracker_fish(img, active_tracker, base):
    # The keypoint indices used for the homography
    # Find the keypoints used in the homography
    eye_idx = m.LABEL_NAMES.index('eye')
    rpec_idx = m.LABEL_NAMES.index('rpec')
    hbi_idx = m.LABEL_NAMES.index('head_body_intercept')
    dfin_idx = m.LABEL_NAMES.index('dfin')
    kp_for_hg = [eye_idx, rpec_idx, hbi_idx, dfin_idx]

    imgs = []
    box_counts = []
    all_kp_warped_and_cropped = []
    for fish_idx in range(len(active_tracker)):
        # find swimming direction, and determine target keypoint set
        eye_x = active_tracker[fish_idx]['keypoints'][-1][2][0]
        dfin_x = active_tracker[fish_idx]['keypoints'][-1][6][0]
        bbox = active_tracker[fish_idx]['last_bbox']
        if eye_x > dfin_x:
            kp_base = base['kp_base_sr']
            bbox_base = base['bbox_base_sr']
            dir = 'sr'
        else:
            kp_base = base['kp_base_sl']
            bbox_base = base['bbox_base_sl']
            dir = 'sl'

        # Format keypoints
        kp = np.array(active_tracker[fish_idx]['keypoints'][-1])[:,0:2]
        kp_hg = kp[kp_for_hg,0:2].astype(np.float32)

        # Find homography
        h, status = cv2.findHomography(kp_hg, kp_base)

        #try:
        # Apply homography to frame and keypoints
        img_warped = cv2.warpPerspective(img.copy(), h, (img.shape[1], img.shape[0]))
        kp_warped = cv2.perspectiveTransform(kp[np.newaxis,:,:], h)

        # Crop warped image according to warped keypoints
        if dir == 'sl':
            img_warped_and_kp_cropped = img_warped[int(kp_warped[0][dfin_idx][1]):int(kp_warped[0][rpec_idx][1]), int(kp_warped[0][eye_idx][0]):int(kp_warped[0][dfin_idx][0])]
            kp_warped_and_cropped = kp_warped.copy()
            kp_warped_and_cropped[0,:,0] = kp_warped_and_cropped[0,:,0]-int(bbox_base[0])
            kp_warped_and_cropped[0,:,1] = kp_warped_and_cropped[0,:,1]-int(kp_warped_and_cropped[0][6][1])
        if dir == 'sr':
            img_warped_and_kp_cropped = img_warped[int(kp_warped[0][dfin_idx][1]):int(kp_warped[0][rpec_idx][1]), int(kp_warped[0][dfin_idx][0]):int(kp_warped[0][eye_idx][0])]
            kp_warped_and_cropped = kp_warped.copy()
            kp_warped_and_cropped[0,:,0] = kp_warped_and_cropped[0,:,0]-int(kp_warped_and_cropped[0][6][0])
            kp_warped_and_cropped[0,:,1] = kp_warped_and_cropped[0,:,1]-int(kp_warped_and_cropped[0][6][1])

        imgs.append(img_warped_and_kp_cropped)
        box_counts.append(active_tracker[fish_idx]['box_count'])
        all_kp_warped_and_cropped.append(kp_warped_and_cropped)

    return imgs, box_counts, all_kp_warped_and_cropped




#img_warped_and_box_cropped = img_warped[int(bbox_base[1]):int(bbox_base[3]), int(bbox_base[0]):int(bbox_base[2])]
#img_cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
#img_warped_and_box_cropped = img_warped[int(bbox_base[1]):int(bbox_base[3]), int(bbox_base[0]):int(bbox_base[2])]
#img_cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
#img_warped_and_kp_cropped = img_warped[int(kp_warped[0][4][1]):int(bbox_base[3]), int(bbox_base[0]):int(kp_warped[0][4][0])]
#img_warped_and_kp_cropped = img_warped[int(kp_warped[0][4][1]):int(bbox_base[2]), int(kp_warped[0][4][0]):int(bbox_base[3])]