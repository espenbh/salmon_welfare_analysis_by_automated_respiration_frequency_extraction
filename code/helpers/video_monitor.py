# This file implements a Tracker class that holds information pertaining to a single salmon, as well as a VideoMonitor class that implements logic for assigning and remove trackers.
# Each Tracker instance contains a Kalman filter that aims to follow the eye of an underlying fish.

import numpy as np
from scipy.optimize import linear_sum_assignment
from helpers.draw_utils import draw_cross
import cv2
import torchvision
from helpers import m
from helpers.kalman import KalmanFilter
from helpers.torax_transform_utils import extract_target_transformation
from helpers.formatting_functions import cv2_to_dot_network_in, cv2_to_RGB_network_in
from helpers.network_call_utils import torax_imgs_to_box_map
import colorsys
import math
import matplotlib.pyplot as plt

class Tracker:
    def __init__(self, id, state, kalman, last_bbox, last_keypoints):
        # For tracking
        self.id = id
        self.state = state
        self.kalman = kalman
        self.consecutive_unmatched = 0

        # For drawing
        self.last_bbox = last_bbox
        self.last_keypoints = last_keypoints

        # For post run inference
        self.full_dists = []
        self.keypoints = []
        self.warped_keypoints = []
        self.warped_dists = []
        self.ang_dists = []
        self.frames = []
        self.clses = []


# Each tracker has one kalman filter, containing the position and position uncertainty of the dot.
# Assign new observations to the most likely tracker
# max_dist: The maximum distance to accept a match between a tracker and a detection
# max_frames_lost: The length of survival for unmatched trackers
# kalman_mod_unc: Model uncertainty for the Kalman filter
# kalman_meas_unc: Measurement uncertainty for the Kalman filter
class VideoMonitor():
    def __init__(self, torax_ID_model, max_dist = 100, max_frames_lost = 50, kalman_mod_unc = 0.5, kalman_meas_unc = 0.01):
        # Lists to store trackers
        self.active_trackers = []
        self.log_trackers = []

        # Tracker parameters
        self.max_dist = max_dist
        self.max_frames_lost = max_frames_lost
        self.id = 0
        self.kalman_mod_unc = kalman_mod_unc
        self.kalman_meas_unc = kalman_meas_unc

        # Classification model
        self.torax_ID_model = torax_ID_model

        # Call the function that generates a keypoint baseline
        exmp_img = 'frame1500'
        fish_idx = 0
        self.base = extract_target_transformation(exmp_img, fish_idx, r'C:\\Users\\espebh\\Documents\\Thesis\\data\\keyrcnn_200_dataset\\manually annotated data')

    def update_trackers(self, model_output, NN_iou_threshold = 0.1):
        # Determine which detections are valid based on nms
        nms = torchvision.ops.nms(model_output['boxes'], model_output['scores'], iou_threshold = NN_iou_threshold)
        nms = nms.detach().cpu().numpy()
        boxes = model_output['boxes'][nms].detach().cpu().numpy()
        keypoints = model_output['keypoints'][nms].detach().cpu().numpy()
        eye_idx = m.LABEL_NAMES.index('eye')

        # Generate a cost matrix with euclidean distances as costs.
        # Assign matches according to Jonker-Volgenant algorithm
        cost = np.zeros((len(boxes), len(self.active_trackers)))
        for i in range(len(boxes)):
            for j in range(len(self.active_trackers)):
                tracker_state = [self.active_trackers[j].state[0].astype(int).item(), self.active_trackers[j].state[1].astype(int).item()]
                cost[i, j] = np.linalg.norm(keypoints[i, eye_idx, 0:2]-tracker_state)
        det_idces, tck_idces = linear_sum_assignment(cost)

        # Update trackers
        for tck_idx in reversed(range(len(self.active_trackers))):
            # Propagate the Kalman filter of all trackers
            pred_state = self.active_trackers[tck_idx].kalman.predict()

            # If the tracker has a match that is close enough
            if tck_idx in tck_idces and cost[det_idces[list(tck_idces).index(tck_idx)], tck_idx] < self.max_dist:
                det_idx = det_idces[list(tck_idces).index(tck_idx)]
                self.active_trackers[tck_idx].last_bbox = boxes[det_idx]
                self.active_trackers[tck_idx].last_keypoints = keypoints[det_idx, :, 0:2]
                self.active_trackers[tck_idx].consecutive_unmatched = 0

                eye = keypoints[det_idx, eye_idx, 0:2]
                upd_state = self.active_trackers[tck_idx].kalman.update(np.matrix([[eye[0]], [eye[1]]]))
                self.active_trackers[tck_idx].state = upd_state
                tck_idx = tck_idx + 1

            # If the tracker does not have a match
            else:
                self.active_trackers[tck_idx].consecutive_unmatched = self.active_trackers[tck_idx].consecutive_unmatched + 1
                self.active_trackers[tck_idx].state = pred_state

                # If the tracker is without matches for max_frames_lost
                if self.active_trackers[tck_idx].consecutive_unmatched > self.max_frames_lost:
                    self.log_trackers.append(self.active_trackers[tck_idx])
                    del self.active_trackers[tck_idx]
                    

        # If the detection is not assigned to a tracker, generate a new tracker
        for det_idx in range(len(nms)):
            if det_idx not in det_idces:
                eye = keypoints[det_idx, eye_idx, 0:2]
                tracker = Tracker(  self.id, 
                                    np.matrix([[eye[0]], [eye[1]]]), 
                                    KalmanFilter(1/m.FRAMES_PER_SECOND, eye[0], eye[1], self.kalman_mod_unc, self.kalman_meas_unc), 
                                    boxes[det_idx], 
                                    keypoints[det_idx, :, 0:2])
                self.active_trackers.append(tracker)
                self.id = self.id + 1

    def update_classes_and_dists(self, img_cv2, frame):
        ljaw_idx = m.LABEL_NAMES.index('ljaw')
        ujaw_idx = m.LABEL_NAMES.index('ujaw')
        rjaw_idx = m.LABEL_NAMES.index('rjaw')

        # Generate salmon torax images
        imgs, IDs, warped_dists, warped_kps = self.warp_active_tracker_fish(img_cv2.copy(), self.base)

        # Find classes of the trackers
        box_map = torax_imgs_to_box_map(imgs, IDs, self.torax_ID_model)

        # Add information about classes and mouth opening to the trackers if a valid class is found
        for i in range(len(self.active_trackers)):
            if self.active_trackers[i].id in list(box_map.keys()): # and box_map[self.active_trackers[i].id][1] > 0:
                ljaw = self.active_trackers[i].last_keypoints[ljaw_idx]
                ujaw = self.active_trackers[i].last_keypoints[ujaw_idx]
                rjaw = self.active_trackers[i].last_keypoints[rjaw_idx]
                self.active_trackers[i].full_dists.append(np.linalg.norm(ujaw-ljaw))
                self.active_trackers[i].keypoints.append(self.active_trackers[i].last_keypoints)
                self.active_trackers[i].warped_keypoints.append(warped_kps[IDs.index(self.active_trackers[i].id)])
                self.active_trackers[i].ang_dists.append(math.degrees(math.atan2(ujaw[1]-rjaw[1], ujaw[0]-rjaw[0]) - math.atan2(ljaw[1]-rjaw[1], ljaw[0]-rjaw[0])))
                self.active_trackers[i].clses.append(box_map[self.active_trackers[i].id])
                self.active_trackers[i].frames.append(frame)
                self.active_trackers[i].warped_dists.append(warped_dists[IDs.index(self.active_trackers[i].id)])


    def warp_active_tracker_fish(self, img, base):
        # Find the keypoints used in the homography
        eye_idx = m.LABEL_NAMES.index('eye')
        rpec_idx = m.LABEL_NAMES.index('rpec')
        hbi_idx = m.LABEL_NAMES.index('head_body_intercept')
        dfin_idx = m.LABEL_NAMES.index('dfin')
        kp_for_hg_idx = [eye_idx, rpec_idx, hbi_idx, dfin_idx]

        # find the keypoints used in the distance calculation
        ljaw_idx = m.LABEL_NAMES.index('ljaw')
        ujaw_idx = m.LABEL_NAMES.index('ujaw')

        # Initialize output arrays
        imgs = []
        IDs = []
        warped_dists = []
        warped_kps = []

        # Iterate over all trackers
        for tck_idx in range(len(self.active_trackers)):
            if self.active_trackers[tck_idx].consecutive_unmatched > 0:
                continue
            # find swimming direction, and determine target keypoint set
            eye_x = self.active_trackers[tck_idx].last_keypoints[eye_idx][0]
            dfin_x = self.active_trackers[tck_idx].last_keypoints[dfin_idx][0]
            if eye_x > dfin_x:
                kp_base = base['kp_base_sr']
                bbox_base = base['bbox_base_sr']
                dir = 'sr'
            else:
                kp_base = base['kp_base_sl']
                bbox_base = base['bbox_base_sl']
                dir = 'sl'

            # Format keypoints
            kp = np.array(self.active_trackers[tck_idx].last_keypoints)[:,0:2]
            kp_hg = kp[kp_for_hg_idx,0:2].astype(np.float32)

            # Find homography
            h, status = cv2.findHomography(kp_hg, kp_base)

            # Apply homography to frame and keypoints
            try:
                kp_warped = cv2.perspectiveTransform(kp[np.newaxis,:,:], h)
                img_warped = cv2.warpPerspective(img.copy(), h, (img.shape[1], img.shape[0]))

                # Crop warped image according to warped keypoints
                if dir == 'sl':
                    img_warped_and_kp_cropped = img_warped[int(kp_warped[0][dfin_idx][1]):int(kp_warped[0][rpec_idx][1]), int(kp_warped[0][eye_idx][0]):int(kp_warped[0][dfin_idx][0])]
                    kp_warped_and_cropped =kp_warped.copy()
                    kp_warped_and_cropped[0,:,0] = kp_warped_and_cropped[0,:,0]-int(bbox_base[0])
                    kp_warped_and_cropped[0,:,1] = kp_warped_and_cropped[0,:,1]-int(kp_warped_and_cropped[0][6][1])
                    #img_warped_and_box_cropped = img_warped[int(bbox_base[1]):int(bbox_base[3]), int(bbox_base[0]):int(bbox_base[2])]
                if dir == 'sr':
                    img_warped_and_kp_cropped = img_warped[int(kp_warped[0][dfin_idx][1]):int(kp_warped[0][rpec_idx][1]), int(kp_warped[0][dfin_idx][0]):int(kp_warped[0][eye_idx][0])]
                    kp_warped_and_cropped =kp_warped.copy()
                    kp_warped_and_cropped[0,:,0] = kp_warped_and_cropped[0,:,0]-int(kp_warped_and_cropped[0][6][0])
                    kp_warped_and_cropped[0,:,1] = kp_warped_and_cropped[0,:,1]-int(kp_warped_and_cropped[0][6][1])
                    #img_warped_and_box_cropped = img_warped[int(bbox_base[1]):int(bbox_base[3]), int(bbox_base[0]):int(bbox_base[2])]

                # Update output arrays if the warping output is valid
                dist = np.linalg.norm(kp_warped[0][ljaw_idx]-kp_warped[0][ujaw_idx])
                if img_warped_and_kp_cropped.shape[0]*img_warped_and_kp_cropped.shape[1]*img_warped_and_kp_cropped.shape[2] > 0:
                    imgs.append(img_warped_and_kp_cropped)
                    IDs.append(self.active_trackers[tck_idx].id)
                    warped_dists.append(dist)
                    warped_kps.append(kp_warped_and_cropped)
            except Exception as e: 
                print('Could not perform warping. Error: \n')
                print(e)

        return imgs, IDs, warped_dists, warped_kps


    def draw_active_tracker(self, img, classes_map = m.CLASSES_MAP_T9, draw_classes = True):
        if self.torax_ID_model == None:
            draw_classes = False

        # Create color palette
        HSV = [(x/m.NUM_KEYPOINTS, 1, 1) for x in range(m.NUM_KEYPOINTS)]

        # Format image
        img = np.multiply(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)), 255).astype(np.uint8)

        # Draw all active trackers
        for instance in self.active_trackers:
            if instance.consecutive_unmatched == 0:
                cv2.rectangle(img, 
                    tuple(instance.last_bbox[:2].astype(int)),
                    tuple(instance.last_bbox[2:].astype(int)), 
                    (255,0,0), 2)
                for keypoint in range(m.NUM_KEYPOINTS):
                    r, g, b = colorsys.hsv_to_rgb(*HSV[keypoint])
                    cv2.circle(img, 
                                (int(instance.last_keypoints[keypoint][0]), int(instance.last_keypoints[keypoint][1])), 2,
                                (255*r, 255*g, 255*b), 2)
                # Draw classes on frame if there is a class classification for this frame
                if draw_classes == True and len(instance.clses) > 0:
                    c = instance.clses[-1][0]
                    cert = instance.clses[-1][1]
                    name = classes_map[int(c)]
                    cv2.putText(img, str(name[:-3]) + ' with cert. ' + str(round(cert, 4)), tuple(instance.last_bbox[:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    #cv2.putText(img, str(name[:-3]), tuple(instance.last_bbox[:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            filter_pos = (instance.state[0].astype(int).item(), instance.state[1].astype(int).item())
            draw_cross(img, filter_pos)
            cv2.putText(img, str(instance.id), filter_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

