import multiprocessing
import os
import glob
import logging

path = 'results/'
outpath = 'results/model1/'

filestoremove = glob.glob(outpath + '/vein*.txt')

for f in filestoremove:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))


def create_vein_model_3D(imagename, outpath, model):
    import cv2
    import numpy as np
    import math

    image_oryg = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)

    image_oryg = np.where(image_oryg == 255, 100, 0).astype('uint8')

    letter = imagename[-5]

    layer = int(imagename[-9:-5])-1554

    if letter == 'b':
        layer = layer + 0.33
    elif letter == 'c':
        layer = layer + 0.66

    # img1 = cv2.Canny(image_oryg, 80, 100)

    mask = np.zeros([600, 390])
    cnts = cv2.findContours(image_oryg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for cnt in cnts:
        if len(cnt) > 0:
            cv2.drawContours(mask, [cnt], -1, 255, 1)

    points_whole_arm = []
    points_whole_arm_seg = []

    points_list = np.where(mask == 255)

    cv2.imwrite('results/' + model + '_cont/' + 'avf' +
                imagename[-9:-5] + letter + '.jpeg', mask.astype("uint8"))

    for i in range(len(points_list[0])):
        points_whole_arm.append([points_list[0][i], points_list[1][i]])

    if len(points_whole_arm) > 0:
        actual_point = points_whole_arm.pop(1)
        first_point = actual_point.copy()
        points_whole_arm_seg.append(actual_point)

        while len(points_whole_arm) > 0:
            distance = []
            for point in points_whole_arm:
                distance.append(
                    math.sqrt((actual_point[0]-point[0])**2) +
                    ((actual_point[1]-point[1])**2))
            index = distance.index(min(distance))
            actual_point = points_whole_arm.pop(index)
            points_whole_arm_seg.append(actual_point)

        points_whole_arm_seg.append(first_point)

        file1 = open(outpath + model + '.txt', 'a')

        for point in range(0, len(points_whole_arm_seg)):
            str1 = str((points_whole_arm_seg[point][0])*0.33) + ';' +\
                str((points_whole_arm_seg[point][1])*0.33) + ';' +\
                str(layer) + '\n'
            file1.write(str1)
        file1.write("\n")
        file1.close()


def process_stage4(image_folder, model_name):
    img_list = glob.glob(image_folder + '/*.png')
    img_list.sort()
    image_num = 1
    for image in img_list:
        logging.info("Process ID %s, %s %i:%i", os.getpid(),
                     image, image_num, len(img_list))
        image_num = image_num + 1
        create_vein_model_3D(image, outpath, model_name)


if __name__ == "__main__":

    formatINFO = "%(asctime)s: %(message)s"
    logging.basicConfig(format=formatINFO,
                        level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("Start creating model 3D")

    process_stage4(path + 'vein_all', 'vein_all')