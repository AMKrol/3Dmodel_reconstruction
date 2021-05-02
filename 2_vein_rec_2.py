# laczenie poszczegolnych zyl w jeden obrazek

import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import logging
import sys

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    formatINFO = "%(asctime)s: %(message)s"
    logging.basicConfig(format=formatINFO,
                        level=logging.INFO, datefmt="%H:%M:%S")
    path = 'results/'

    folders = ['vein_0', 'vein_1', 'vein_2', 'vein_3', 'vein_4']
    mypath = path+folders[0]
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()

    size = (390, 600)
    out = cv2.VideoWriter('results/tracker.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for filename in onlyfiles:
        image_save = np.zeros([600, 390])

        for veindir in folders:
            img_step = cv2.imread(path + veindir + '/' +
                                  filename, cv2. IMREAD_GRAYSCALE)
            img_step2 = np.where(img_step == 255, 1, 0).astype('uint8')

            image_save = np.where(
                img_step2 == 1, image_save+img_step2, image_save)
            # for r in range(0,599):
            #    for c in range(0,389):
            #        image_save[r,c] = image_save[r,c] + img_step2[r,c]

        # cv2.imshow("test", image_save*255)
        # cv2.waitKey(50)
        cv2.imwrite('results/vein_all/' + filename, image_save*255)
        close = image_save*255
        frame_step3 = np.repeat(
                close[:, :, np.newaxis], repeats=3, axis=2).astype('uint8')
        out.write(frame_step3)

    out.release()
