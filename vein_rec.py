import cv2
import numpy as np
import glob
import math

# Using readlines()
model_name = 'vein_0'
file1 = open(model_name + '.txt', 'r')
Lines = file1.readlines()
outpath = 'results/'

# Strips the newline character
angle = 0
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

for line in Lines:

    line2 = line.split()
    name = line2[0][5:9]
    letter = line2[0][9]
    X = line2[1].replace(',', '')
    Y = line2[2].replace(']', '')

    level = int(name)-1554
    if letter == 'b':
        level = level + 0.33
    elif letter == 'c':
        level = level + 0.66

    image = np.zeros([600, 390])
    coord = (int(X), int(Y))
    axesLength = (3, 4)
    image2 = cv2.ellipse(image, coord, axesLength,  angle, 0, 360, 255, -1)

    whitepoints = np.where(image2 == 255)

    points = []
    for i in range(len(whitepoints[0])):
        points.append((whitepoints[1][i], whitepoints[0][i]))

    temp = cv2.convexHull(np.int32(points))

    hull_list = []
    hull_list.append(temp)

    mask_inner = np.zeros([600, 390])
    cv2.drawContours(mask_inner, hull_list, 0, 1, 1)
    mask_inner = mask_inner.astype('uint8')

    white_points = np.where(mask_inner == 1)

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

    points_whole_arm_seg = []

    actual_point = points_whole_arm.pop(1)
    first_point = actual_point.copy()
    points_whole_arm_seg.append(actual_point)

    while len(points_whole_arm) > 0:
        distance = []
        for point in points_whole_arm:
            distance.append(math.sqrt(
                (actual_point[0] - point[0]) ** 2) + ((actual_point[1] - point[1]) ** 2))
            index = distance.index(min(distance))
            actual_point = points_whole_arm.pop(index)
            points_whole_arm_seg.append(actual_point)

    points_whole_arm_seg.append(first_point)

    file1 = open(outpath + 'modelV2/' + model_name + '.txt', 'a')

    for point in range(0, len(points_whole_arm_seg)):
        str1 = str((points_whole_arm_seg[point][0]) * 0.33) + ';' \
            + str((points_whole_arm_seg[point][1]) * 0.33) \
            + ';' + str(layer) + '\n'
        file1.write(str1)
    file1.write("\n")
    file1.close()

    #cv2.imshow("window_name", mask_inner*255)
    # cv2.waitKey(20)
    angle += 1
    print("  ")
    print(level)
    print(X)
    print(Y)
    print(angle_x)
    print(angle_y)
    print(z_image)
