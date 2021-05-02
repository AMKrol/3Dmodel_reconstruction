import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import logging

# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

if __name__ == '__main__':
    formatINFO = "%(asctime)s: %(message)s"
    logging.basicConfig(format=formatINFO,
                        level=logging.INFO, datefmt="%H:%M:%S")
    mypath = 'results/1_outer'

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()

    # Read video
    frame = cv2.imread(mypath + '/' + onlyfiles[0], cv2.IMREAD_COLOR)

    # Define an initial bounding box
    bbox = [[(80, 90, 30, 30), "CSRT"],
            [(71, 141, 22, 24), "CSRT"],
            [(230, 300, 30, 30), "CSRT"],
            [(175, 215, 30, 30), "CSRT"],
            [(117, 230, 40, 40), "CSRT"]]  # ,
    # [(135, 250, 40, 37), "CSRT"]]

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)
    height, width, layers = frame.shape
    size = (width, height)
    out = cv2.VideoWriter('results/tracker.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    outtest = cv2.VideoWriter(
        'results/trackertest.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    out.write(frame)
    # Initialize tracker with first frame and bounding box
    trackers = []

    for i in range(len(bbox)):

        tracker_type = bbox[i][1]

        if tracker_type == 'BOOSTING':
            trackers.append(cv2.TrackerBoosting_create())
        if tracker_type == 'MIL':
            trackers.append(cv2.TrackerMIL_create())
        if tracker_type == 'KCF':
            trackers.append(cv2.TrackerKCF_create())
        if tracker_type == 'TLD':
            trackers.append(cv2.TrackerTLD_create())
        if tracker_type == 'MEDIANFLOW':
            trackers.append(cv2.TrackerMedianFlow_create())
        if tracker_type == 'GOTURN':
            trackers.append(cv2.TrackerGOTURN_create())
        if tracker_type == 'MOSSE':
            trackers.append(cv2.TrackerMOSSE_create())
        if tracker_type == "CSRT":
            trackers.append(cv2.TrackerCSRT_create())

        trackers[i].init(frame, bbox[i][0])

    for i in range(0, len(onlyfiles)):

        logging.info("%i:%i", i+1, len(onlyfiles))

        # Read a new frame
        frame_oryg = cv2.imread(mypath + '/' + onlyfiles[i], cv2.IMREAD_COLOR)
        frame_save = frame_oryg.copy()

        mask_veins = np.zeros([height, width])

        frame_track = np.zeros([height, width, 3])
        # Update tracker
        for j in range(len(trackers)):

            tracker = trackers[j]
            mask = np.zeros([height, width])
            ok, bbox = tracker.update(frame_oryg)

            frame_truck_step = np.zeros([height, width])
            # Draw bounding box
            if ok:
                # Tracking success

                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                cv2.rectangle(frame_save, p1, p2, (255, 0, 0), 2, 1)

                cv2.rectangle(mask, p1, p2, 1, -1, 1)
                frame_step = cv2.cvtColor(frame_oryg, cv2.COLOR_BGR2GRAY)

                frame_step = np.where(mask == 1, frame_step, 0).astype("uint8")

                frame_step3 = cv2.equalizeHist(frame_step)

                if j == 0:
                    ret, frame_step3 = cv2.threshold(
                        frame_step3, 20, 255, cv2.THRESH_BINARY)

                elif j == 1:
                    ret, frame_step3 = cv2.threshold(
                        frame_step3, 30, 255, cv2.THRESH_BINARY)

                elif j == 2:
                    ret, frame_step3 = cv2.threshold(
                        frame_step3, 70, 255, cv2.THRESH_BINARY)

                elif j == 3:
                    ret, frame_step3 = cv2.threshold(
                        frame_step3, 90, 255, cv2.THRESH_BINARY)

                elif j == 4:
                    ret, frame_step3 = cv2.threshold(
                        frame_step3, 60, 255, cv2.THRESH_BINARY)

            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (0, 550),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

            frame_step3 = np.where(frame_step3 > 1, 255, 0).astype("uint8")

            mask = np.zeros([height + 2, width + 2], np.uint8)
            frame_step3 = frame_step3.astype("uint8")
            cv2.floodFill(frame_step3, mask, (10, 10), 255,
                          1, 1, cv2.FLOODFILL_FIXED_RANGE)
            frame_step3 = np.where(frame_step3 > 1, 0, 255).astype("uint8")

            # Filter using contour area and remove small noise
            cnts = cv2.findContours(
                frame_step3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                area = cv2.contourArea(c)
                if area < 5:
                    cv2.drawContours(frame_step3, [c], -1, (0, 0, 0), -1)

            # Morph close and invert image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            close = 255 - \
                cv2.morphologyEx(frame_step3, cv2.MORPH_CLOSE,
                                 kernel, iterations=2)

            close = np.where(close == 255, 0, 255).astype("uint8")
            frame_step3 = np.repeat(
                close[:, :, np.newaxis], repeats=3, axis=2).astype('uint8')
            frame_track = frame_track + frame_step3

            frame_track = np.where(frame_track > 1, 0, 255).astype("uint8")

            cv2.imwrite('results/vein_' + str(j) +
                        '/' + onlyfiles[i], frame_step3)

        outtest.write(frame_track)

        out.write(frame_save)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    out.release()
    outtest.release()
