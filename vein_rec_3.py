import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import glob
import math

if __name__ == '__main__':
    mypath = 'results/vein_all_cont'

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()

    # Strips the newline character
    angle = 0
    outpath = 'results/'
    model = outpath + '1_outer_mask'
    img_list = glob.glob(model + '/*.png')
    img_list.sort()

    image_first = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
    image_last = cv2.imread(img_list[-1], cv2.IMREAD_GRAYSCALE)

    white_points = np.where(image_first == 255)

    points = []
    for i in range(len(white_points[0])):
        points.append((white_points[1][i], white_points[0][i]))

    [x1, y1, w1, h1] = cv2.boundingRect(np.int32(points))

    white_points = np.where(image_last == 255)

    points = []
    for i in range(len(white_points[0])):
        points.append((white_points[1][i], white_points[0][i]))

    [x2, y2, w2, h2] = cv2.boundingRect(np.int32(points))
    z2 = int(img_list[-1][-9:-5])-1554
    letter = img_list[-1][-5]
    if letter == 'b':
        z2 = z2 + 0.33
    elif letter == 'c':
        z2 = z2 + 0.66

    center_first = np.int32((x1+w1/2, y1+h1/2))
    center_last = np.int32((x2+w2/2, y2+h2/2))

    angle_x = math.atan((center_first[0] - center_last[0]) / z2)
    angle_y = math.atan((center_first[1] - center_last[1]) / z2)

    for filename in onlyfiles:
        name = filename[3:7]
        letter = filename[7]

        level = int(name)-1554
        if letter == 'b':
            level = level + 0.33
        elif letter == 'c':
            level = level + 0.66

        image = cv2.imread(mypath + '/' + filename, cv2.IMREAD_GRAYSCALE)

        mask = np.zeros([600, 390])
        cnts = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(mask, [c], -1, 255, 1)

        white_points = np.where(mask == 255)

        points = []
        for i in range(len(white_points[0])):
            points.append((white_points[1][i], white_points[0][i]))

        x_corr = math.tan(angle_x) * level
        y_corr = math.tan(angle_y) * level

        z_image = math.sqrt(x_corr*x_corr + y_corr*y_corr)

        layer = math.sqrt(z_image*z_image + level*level)

        points_whole_arm = []
        for point in points:
            points_whole_arm.append(
                [np.float(point[0]) + x_corr, np.float(point[1]) + y_corr])

        file1 = open(outpath + 'modelV2/test2.txt', 'a')

        for point in range(0, len(points_whole_arm)):
            str1 = str((points_whole_arm[point][0]) * 0.33) + ';' + str((points_whole_arm[point][1]) * 0.33) \
                + ';' + str(layer) + '\n'
            file1.write(str1)
        file1.write("\n")
        file1.close()
