import cv2
import numpy as np
import os

TP = 0
TN = 0
FP = 0
FN = 0

# - THRESHOLDs:
#   - pedestrains = 10
#   - highway = 9
#   - office = 1 (not a lot of movement, so subtracting two frames does not give good results)
# - Other info:
#   - run office sequence from 510 frame

THRESHOLD = 15
FRAME_BUFFER_SIZE = 60

DIR = os.getcwd() + '/../sequences'
SEQUENCE = '/office'
DIR = DIR + SEQUENCE


def mean_bg(frame_buffer):
    bg_mean = np.mean(frame_buffer, axis=2)
    bg_mean = np.uint8(bg_mean)
    return bg_mean


def mean_bg_conservative(frame_buffer, prev_bg, prev_mask):
    binary_mask_bg = np.uint8(prev_mask == 0)
    binary_mask_fg = np.uint8(prev_mask == 255)
    bg_mean_temp = np.mean(frame_buffer, axis=2)
    bg_mean_temp = np.uint8(bg_mean_temp)
    bg_mean = np.multiply(binary_mask_bg, bg_mean_temp) + np.multiply(binary_mask_fg, prev_bg)
    return bg_mean


def median_bg(frame_buffer):
    bg_median = np.median(frame_buffer, axis=2)
    bg_median = np.uint8(bg_median)
    return bg_median


def median_bg_conservative(frame_buffer, prev_bg, prev_mask):
    binary_mask_bg = np.uint8(prev_mask == 0)
    binary_mask_fg = np.uint8(prev_mask == 255)
    bg_median_temp = np.median(frame_buffer, axis=2)
    bg_median_temp = np.uint8(bg_median_temp)
    bg_median = np.multiply(binary_mask_bg, bg_median_temp) + np.multiply(binary_mask_fg, prev_bg)
    return bg_median


def update_frame_buffer(current_frame, frame_buffer, frame_top_ptr):
    frame_buffer[:, :, frame_top_ptr] = current_frame
    frame_top_ptr = frame_top_ptr + 1
    if frame_top_ptr == 60:
        frame_top_ptr = 0
    return frame_buffer, frame_top_ptr


def read_roi_values():
    f = open(DIR + '/temporalROI.txt', 'r')
    line = f.readline()
    roi_start_file, roi_end_file = line.split()
    roi_start_file = int(roi_start_file)
    roi_end_file = int(roi_end_file)
    return roi_start_file, roi_end_file


if __name__ == '__main__':
    roi_start, roi_end = read_roi_values()

    frame = cv2.imread(DIR+'/input/in000' + str(roi_start) + '.jpg', cv2.IMREAD_GRAYSCALE)

    frames_buffer = np.zeros((frame.shape[0], frame.shape[1], FRAME_BUFFER_SIZE), np.uint8)
    iN = 0

    kernel = np.ones((5, 5), np.uint8)

    for i in range(roi_start, roi_end, 1):
        current_img_color = cv2.imread(DIR+'/input/in%06d.jpg' % i)
        current_img = current_img_color.copy()
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        frames_buffer, iN = update_frame_buffer(current_img, frames_buffer, iN)
        if i < roi_start+60:
            continue

        # bg_mask = median_bg(frames_buffer)
        # bg_mask = mean_bg(frames_buffer)

        if i > roi_start+60+1: # run conservative update when first mask is currently calculated
            # bg_mask = mean_bg_conservative(frames_buffer, bg_mask, mask)
            bg_mask = median_bg_conservative(frames_buffer, bg_mask, mask)
        else:
            # bg_mask = mean_bg(frames_buffer)
            bg_mask = median_bg(frames_buffer)

        cv2.imshow('Bg mask', bg_mask)

        diff = cv2.absdiff(current_img, bg_mask)
        diff = cv2.medianBlur(diff, 5)

        cv2.imshow('Diff', diff)

        mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

        # cv2.imshow('Mask', mask)

        mask = cv2.medianBlur(mask, 5)
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)

        # cv2.imshow('Morphed mask', mask)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        # cv2.imshow("Labels", np.uint8(labels / stats.shape[0] * 255))

        if stats.shape[0] > 1:      # czy sa jakies obiekty
            tab = stats[1:,4]       # wyciecie 4 kolumny bez pierwszego elementu
            pi = np.argmax( tab )   # znalezienie indeksu najwiekszego elementu
            pi = pi + 1             # inkrementacja bo chcemy indeks w stats, a nie w tab

            # wyrysownie bbox
            cv2.rectangle(img=current_img_color,
                          pt1=(stats[pi, 0], stats[pi, 1]),
                          pt2=(stats[pi, 0]+stats[pi, 2], stats[pi, 1]+stats[pi, 3]),
                          color=(0, 0, 255),
                          thickness=2)
            cv2.circle(img=current_img_color,
                       center=(np.int(centroids[pi, 0]), np.int(centroids[pi, 1])),
                       radius=2,
                       color=(0, 255, 0),
                       thickness=3)

            # wypisanie informacji o polu i numerze najwiekszego elementu
            cv2.putText(img=current_img_color,
                        text="%d" % stats[pi, 4],
                        org=(stats[pi, 0], stats[pi, 1]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0))
            cv2.putText(img=current_img_color,
                        text="%d" % pi,
                        org=(np.int(centroids[pi, 0]), np.int(centroids[pi, 1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0,255,0))

        # cv2.imshow('Labeled frame', current_img_color)

        # Evaluate
        gt_img = cv2.imread(DIR+'/groundtruth/gt%06d.png' % i)

        # cv2.imshow('gt', gt_img)

        TP_M = np.logical_and((mask == 255), (gt_img[:, :, 0] == 255))
        TP_S = np.sum(TP_M)

        FP_M = np.logical_and((mask == 255), (gt_img[:, :, 0] == 0))
        FP_S = np.sum(FP_M)

        FN_M = np.logical_and((mask == 0), (gt_img[:, :, 0] == 255))
        FN_S = np.sum(FN_M)

        TP = TP + TP_S
        FP = FP + FP_S
        FN = FN + FN_S

        if cv2.waitKey(1) == 27:
            break

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2*P*R/(P + R)

    print('P:', P)
    print('R:', R)
    print('F1:', F1)

