# This file has a movie file as input, and saves every tenth frame in the movie file to disk.

import cv2
start_frame = 11000
stop_frame = 15000
vidcap = cv2.VideoCapture('C:\\Users\\espebh\\Documents\\Thesis\\data\\Annotations keyrcnn industry net\\videos\\flipped_GOPR9772.avi')
success, image = vidcap.read()
count = start_frame
vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
while success:
    if not count % 10:
        cv2.imwrite("C:\\Users\\espebh\\Documents\\Thesis\\data\\Annotations keyrcnn industry net\\images\\frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    count += 1
    if not count % 100:
        print('Frame nr: ', count)
    if count > stop_frame:
        break


vidcap.release()  
