import torchvision
import cv2

# Calculate intersection over union for two bounding boxes
def bb_intersection_over_union(boxA, boxB):
    # Each bbox: xmin, ymin, xmax, ymax
    # Logic is copied from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    assert boxA[0] < boxA[2]
    assert boxA[1] < boxA[3]
    assert boxB[0] < boxB[2]
    assert boxB[1] < boxB[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0

    return iou

# Function to update trackers
# Active_tracker is a list of dictionaries. Each dictionary is matching one fish. [{last_bbox: [4], last_keypoints = [3], dists[n], first_frame: c, last_frame: c, box_number: c}]
# Log_tracker is a list of all active_trackers that lost its fish.
# Last_bbox: the coordinates of the last registered bounding box of the fish
# Last_keypoints: The coordinates of the last registered keypoint of the fish
# Keypoints: A timesequence of euclidean distances between the lower and upper jaw of the fish.
# First_frame and last_frame: The first and last frame this fish is registered
# Box_count: The number of fish detected before this one.
# NN_iou_treshold: Percentage of overlap we allow between boxes of one frame
# track_iou_treshold: Percentage of overlap we require for a box in two frames to be mapped to the same fish
# targets is the output of a neural network: [{'boxes: tensor([[]]), 'scores': tensor([]), keypoints: tensor([[[]]])}]
def update_trackers(active_tracker, log_tracker, frame, targets, box_count, img, NN_iou_threshold = 0.1, track_iou_treshold = 0.1):
    # We check all frames in the batch
    for idx in range(len(targets)):
        # Determine which bboxes are valid based on nms
        nms = torchvision.ops.nms(targets[idx]['boxes'], targets[idx]['scores'], iou_threshold = NN_iou_threshold)
        nms = nms.detach().cpu().numpy()
        boxes = targets[idx]['boxes'].detach().cpu().numpy()
        keypoints = targets[idx]['keypoints'].detach().cpu().numpy()
        img = cv2.resize(img, (1600, 800), interpolation=cv2.INTER_AREA)

        # Check all fishes registered in the current frame, starting with the most certain fish. 
        for i in nms:
            hit = 0
            # Check if any of the active trackers have sufficiently overlap with the detected fish.
            # If it does, this active tracker is updated.
            # No active tracker can be updated twice in the same frame, this is the role of the last if check.
            for ab in range(len(active_tracker)):
                if bb_intersection_over_union(active_tracker[ab]['last_bbox'], boxes[i]) > track_iou_treshold and active_tracker[ab]['last_frame'] != frame:
                    active_tracker[ab]['keypoints'].append(keypoints[i])
                    active_tracker[ab]['last_bbox'] = boxes[i]
                    active_tracker[ab]['last_frame'] = frame
                    hit = 1
                    break
            # If no active tracker match the detected fish, a new active tracker is created
            if hit == 0:
                new_inst = {'last_bbox': boxes[i], 
                    'keypoints': [keypoints[i]], 
                    'first_frame': frame, 
                    'last_frame': frame, 
                    'box_count': box_count}
                active_tracker.append(new_inst)
                box_count = box_count + 1
        # Find the trackers not updated in this frame
        keep_idx = []
        for ab in range(len(active_tracker)):
            # If the active tracker is updated in this frame, add its index to keep_idx
            if active_tracker[ab]['last_frame'] == frame:
                keep_idx.append(ab)
            else:
            # If the active tracker is not updated in this frame, the fish is gone, and the tracker is logged
                log_tracker.append(active_tracker[ab])
        active_tracker = [active_tracker[j] for j in keep_idx]
    return active_tracker, log_tracker, box_count


