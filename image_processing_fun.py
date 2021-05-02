def outer_analysis(mypath, imagename, outpath):
    import cv2
    import numpy as np
    from copy import copy

    image_oryg = cv2.imread(mypath + '/' + imagename, cv2.IMREAD_COLOR)

    img = image_oryg[260:860, 1500:1890]

    z = img.reshape((-1, 3))
    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    k = 6
    ret, label, center = cv2.kmeans(
        z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img_fragment = res.reshape(img.shape)

    cv2.imwrite(outpath + '0_color/' + imagename, img_fragment)

    mask_pusta = np.zeros(img_fragment.shape[:2], np.uint8)

    fgdModel_fragment = np.zeros((1, 65), np.float64)
    bgdModel_fragment = np.zeros((1, 65), np.float64)

    rect = (40, 140, 310, 465)
    mask_start = copy(mask_pusta)
    cv2.grabCut(img_fragment, mask_start, rect, bgdModel_fragment,
                fgdModel_fragment, 20, cv2.GC_INIT_WITH_RECT)

    mask_outer = np.where((mask_start == 2) | (
        mask_start == 0), 0, 1).astype('uint8')

    kernel = np.ones((5, 5), np.uint8)

    mask_outer = cv2.dilate(mask_outer, kernel, iterations=1)
    mask_outer = cv2.erode(mask_outer, kernel, iterations=1)

    mask_outer = np.where(mask_outer == 1, 255, 0).astype('uint8')
    mask_floodfill = mask_outer.copy()
    h, w = mask_outer.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_floodfill, mask, (0, 0), 255)
    mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)

    mask_outer = mask_outer | mask_floodfill_inv
    mask_outer = cv2.erode(mask_outer, kernel, iterations=3)
    mask_outer = cv2.dilate(mask_outer, kernel, iterations=3)
    mask_outer = np.where(mask_outer == 255, 1, mask_outer).astype('uint8')

    img_outer = img * mask_outer[:, :, np.newaxis]

    img_out = np.repeat(mask_outer[:, :, np.newaxis], 3, axis=2)

    cv2.imwrite(outpath + '1_outer_mask/' + imagename, img_out * 255)
    cv2.imwrite(outpath + '2_outer/' + imagename, img_outer)

    return mask_outer, fgdModel_fragment, bgdModel_fragment


def outer_analysis_step(mypath, imagename, outpath, parametry):
    import cv2
    import numpy as np

    mask_outer_temp = parametry[0].copy()
    fgdModel_fragment = parametry[1].copy()
    bgdModel_fragment = parametry[2].copy()

    image_oryg = cv2.imread(mypath + '/' + imagename, cv2.IMREAD_COLOR)

    img = image_oryg[260:860, 1500:1890]

    kernel = np.ones((5, 5), np.uint8)
    mask_outer_diff_dilatation_step = cv2.dilate(
        mask_outer_temp, kernel, iterations=1) - mask_outer_temp
    mask_outer_erode_step = cv2.erode(mask_outer_temp, kernel, iterations=1)
    mask_outer_diff_erode_step = mask_outer_temp - mask_outer_erode_step

    mask_outer_step = np.ones(mask_outer_temp.shape, np.uint8)
    mask_outer_step = mask_outer_step * cv2.GC_BGD

    for i in range(len(mask_outer_temp)):
        for j in range(len(mask_outer_temp[0])):
            mask_outer_step[i][j] = \
                mask_outer_erode_step[i][j] * cv2.GC_FGD + \
                mask_outer_diff_erode_step[i][j] * cv2.GC_PR_FGD + \
                mask_outer_diff_dilatation_step[i][j] * cv2.GC_PR_BGD

    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 6
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img_fragment = res.reshape(img.shape)

    cv2.imwrite(outpath + '0_color/' + imagename, img_fragment)

    cv2.grabCut(img_fragment, mask_outer_step, None, bgdModel_fragment,
                fgdModel_fragment, 20, cv2.GC_INIT_WITH_MASK)

    mask_outer = np.where((mask_outer_step == 2) | (
        mask_outer_step == 0), 0, 1).astype('uint8')
    img_outer = img * mask_outer[:, :, np.newaxis]

    mask_outer_save = np.where(mask_outer == 1, 255, 0).astype('uint8')

    cv2.imwrite(outpath + '1_outer_mask/' + imagename, mask_outer_save)
    cv2.imwrite(outpath + '2_outer/' + imagename, img_outer)

    return mask_outer, fgdModel_fragment, bgdModel_fragment


def inner_analysis(mypath, imagename, outpath):
    import cv2
    import numpy as np

    image_oryg = cv2.imread(mypath + '/' + imagename, cv2.IMREAD_COLOR)

    z = image_oryg.reshape((-1, 3))
    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    k = 6
    ret, label, center = cv2.kmeans(
        z, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img_inner = res.reshape(image_oryg.shape)

    img_gray = cv2.cvtColor(img_inner, cv2.COLOR_BGR2GRAY)

    unique_list = []
    img_gray2 = np.reshape(img_gray, (-1, 1))
    img_gray2 = img_gray2.tolist()
    # traverse for all elements
    for x in img_gray2:
        if x not in unique_list:
            unique_list.append(x)

    unique_list = np.array(unique_list)

    col_1 = unique_list[(unique_list > 35) & (unique_list < 55)]

    mask_temp = np.where((img_gray == col_1), 255, 0).astype('uint8')

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask_temp, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 100

    mask_temp = np.ones(output.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask_temp[output == i + 1] = 0

    mask_temp = np.where(mask_temp == 1, 255, 0).astype("uint8")

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask_temp, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 100

    mask_temp2 = np.ones(output.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask_temp2[output == i + 1] = 0

    mask_temp2 = mask_temp2.astype('uint8')

    whitepoints = np.where(mask_temp2 == 1)

    points = []
    for i in range(len(whitepoints[0])):
        points.append((whitepoints[1][i], whitepoints[0][i]))

    temp = cv2.convexHull(np.int32(points))
    ellipse = cv2.fitEllipse(temp)

    test_image = np.zeros(image_oryg.shape)

    cv2.ellipse(test_image, ellipse, 255, 1)
    test_image = test_image.astype('uint8')

    whitepoints = np.where(test_image == 255)

    points = []
    for i in range(len(whitepoints[0])):
        points.append((whitepoints[1][i], whitepoints[0][i]))

    temp = cv2.convexHull(np.int32(points))

    hull_list = []
    hull_list.append(temp)

    mask_inner = np.zeros(output.shape)
    cv2.drawContours(mask_inner, hull_list, 0, 1, -1)
    mask_inner = mask_inner.astype('uint8')

    image_inner = image_oryg * mask_inner[:, :, np.newaxis]

    cv2.imwrite(outpath + '4_inner/' + imagename, image_inner)
    cv2.imwrite(outpath + '3_inner_mask/' + imagename, mask_inner * 255)


def bones_analysis(mypath, imagename, outpath):
    import cv2
    import numpy as np

    image_oryg = cv2.imread(mypath + '/' + imagename, cv2.IMREAD_COLOR)

    z = image_oryg.reshape((-1, 3))
    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    k = 5
    ret, label, center = cv2.kmeans(
        z, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img_bones = res.reshape(image_oryg.shape)

    img_gray = cv2.cvtColor(img_bones, cv2.COLOR_BGR2GRAY)

    unique_list = []
    img_gray2 = np.reshape(img_gray, (-1, 1))
    img_gray2 = img_gray2.tolist()
    # traverse for all elements
    for x in img_gray2:
        if x not in unique_list:
            unique_list.append(x)

    unique_list = np.array(unique_list)

    col_1 = unique_list[(unique_list > 150)]

    mask_temp = np.where((img_gray == col_1), 255, 0).astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask_temp, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 70

    mask_temp = np.zeros(output.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask_temp[output == i + 1] = 1

    mask_temp = np.where(mask_temp == 1, 120, 0).astype("uint8")

    canny_output = cv2.Canny(mask_temp, 100, 120)

    mask_empty = np.zeros(canny_output.shape)

    mask_temp1 = cv2.circle(mask_empty.copy(), (235, 270), 35, 1, -1)

    mask_temp2 = cv2.circle(mask_empty.copy(), (180, 380), 35, 1, -1)

    bone_1 = (canny_output * mask_temp1)
    bone_2 = (canny_output * mask_temp2)

    whitepoints1 = np.where(bone_1 == 255)

    points = []
    for i in range(len(whitepoints1[0])):
        points.append((whitepoints1[1][i], whitepoints1[0][i]))

    temp1 = cv2.convexHull(np.int32(points))

    whitepoints2 = np.where(bone_2 == 255)

    points = []
    for i in range(len(whitepoints2[0])):
        points.append((whitepoints2[1][i], whitepoints2[0][i]))

    temp2 = cv2.convexHull(np.int32(points))

    hull_list = [temp1, temp2]

    mask_bone1 = np.zeros(output.shape)
    mask_bone2 = np.zeros(output.shape)
    cv2.drawContours(mask_bone1, hull_list, 0, 255, -1)
    cv2.drawContours(mask_bone2, hull_list, 1, 255, -1)

    mask_temp1 = np.where(mask_bone1 == 255, 1, 0).astype('uint8')
    mask_temp2 = np.where(mask_bone2 == 255, 1, 0).astype('uint8')

    img_bone1 = image_oryg*mask_temp1[:, :, np.newaxis]
    img_bone2 = image_oryg * mask_temp2[:, :, np.newaxis]

    cv2.imwrite(outpath + '5_bone1_mask/' + imagename, mask_bone1)
    cv2.imwrite(outpath + '6_bone1/' + imagename, img_bone1)

    cv2.imwrite(outpath + '7_bone2_mask/' + imagename, mask_bone2)
    cv2.imwrite(outpath + '8_bone2/' + imagename, img_bone2)

    return mask_bone1, mask_bone2


def bones_analysis_step(mypath, imagename, outpath, parametry):
    import cv2
    import numpy as np

    mask1 = parametry[0].copy()
    mask2 = parametry[1].copy()

    image_oryg = cv2.imread(mypath + '/' + imagename, cv2.IMREAD_COLOR)

    z = image_oryg.reshape((-1, 3))
    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    k = 5
    ret, label, center = cv2.kmeans(
        z, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img_bones = res.reshape(image_oryg.shape)

    img_gray = cv2.cvtColor(img_bones, cv2.COLOR_BGR2GRAY)

    unique_list = []
    img_gray2 = np.reshape(img_gray, (-1, 1))
    img_gray2 = img_gray2.tolist()
    # traverse for all elements
    for x in img_gray2:
        if x not in unique_list:
            unique_list.append(x)

    unique_list = np.array(unique_list)

    col_1 = unique_list[(unique_list > 150)]

    mask_temp = np.where((img_gray == col_1), 255, 0).astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask_temp, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 70

    mask_temp = np.zeros(output.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask_temp[output == i + 1] = 1

    mask_temp = np.where(mask_temp == 1, 255, 0).astype("uint8")

    kernel = np.ones((5, 5), np.uint8)
    mask1 = cv2.dilate(mask1, kernel, iterations=1).astype('uint8')
    mask2 = cv2.dilate(mask2, kernel, iterations=1)

    mask1 = np.where(mask1 == 255, 1, 0).astype('uint8')
    mask2 = np.where(mask2 == 255, 1, 0).astype('uint8')

    bone_1 = mask_temp * mask1
    bone_2 = mask_temp * mask2

    whitepoints1 = np.where(bone_1 == 255)

    points = []
    for i in range(len(whitepoints1[0])):
        points.append((whitepoints1[1][i], whitepoints1[0][i]))

    temp1 = cv2.convexHull(np.int32(points))

    whitepoints2 = np.where(bone_2 == 255)

    points = []
    for i in range(len(whitepoints2[0])):
        points.append((whitepoints2[1][i], whitepoints2[0][i]))

    temp2 = cv2.convexHull(np.int32(points))

    hull_list = [temp1, temp2]

    mask_bone1 = np.zeros(output.shape)
    mask_bone2 = np.zeros(output.shape)

    cv2.drawContours(mask_bone1, hull_list, 0, 255, -1)
    cv2.drawContours(mask_bone2, hull_list, 1, 255, -1)

    mask_temp1 = np.where(mask_bone1 == 255, 1, 0).astype('uint8')
    mask_temp2 = np.where(mask_bone2 == 255, 1, 0).astype('uint8')

    img_bone1 = image_oryg * mask_temp1[:, :, np.newaxis]
    img_bone2 = image_oryg * mask_temp2[:, :, np.newaxis]

    cv2.imwrite(outpath + '5_bone1_mask/' + imagename, mask_bone1)
    cv2.imwrite(outpath + '6_bone1/' + imagename, img_bone1)

    cv2.imwrite(outpath + '7_bone2_mask/' + imagename, mask_bone2)
    cv2.imwrite(outpath + '8_bone2/' + imagename, img_bone2)

    return mask_bone1, mask_bone2


def create_model_3D(imagename, outpath, model):
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

    img1 = cv2.Canny(image_oryg, 80, 100)

    points_whole_arm = []
    points_whole_arm_seg = []

    points_list = np.where(img1 == 255)

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


def model_straightening(outpath):

    import os
    import cv2
    import numpy as np
    import glob
    import math
    import multiprocessing

    filestoremove = glob.glob('model3D/*.txt')

    for f in filestoremove:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    jobs = []

    model = [[outpath + '1_outer_mask', 'outer'],
             [outpath + '3_inner_mask', 'inner'],
             [outpath + '5_bone1_mask', 'bone1'],
             [outpath + '7_bone2_mask', 'bone2']]

    img_list = glob.glob(model[0][0] + '/*.png')

    img_list.sort()
    image_first = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
    image_last = cv2.imread(img_list[-1], cv2.IMREAD_GRAYSCALE)

    image_next = cv2.Canny(image_first, 100, 120)

    white_points = np.where(image_next == 255)

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

    print(center_first)
    print(center_last)
    print(z2)

    angle_x = math.atan((center_first[0] - center_last[0]) / z2)
    angle_y = math.atan((center_first[1] - center_last[1]) / z2)

    print(math.degrees(angle_x))
    print(math.degrees(angle_y))

    for i in model:
        img_list = glob.glob(i[0] + '/*.png')
        img_list.sort()
        p = multiprocessing.Process(target=process_stage5, args=(
            img_list, i[1], outpath, angle_x, angle_y))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()


def process_stage5(img_list, model_name, outpath, angle_x, angle_y):
    import logging
    import os
    import cv2
    import numpy as np
    import math

    image_num = 1
    for image in img_list:

        logging.info("Process ID %s, %s %i:%i", os.getpid(),
                     image, image_num, len(img_list))
        image_num = image_num + 1
        image_next = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_next = np.where(image_next == 255, 120, 0).astype('uint8')

        image_next = cv2.Canny(image_next, 100, 120)

        white_points = np.where(image_next == 255)

        points = []
        for i in range(len(white_points[0])):
            points.append((white_points[1][i], white_points[0][i]))

        z_next = int(image[-9:-5]) - 1554
        letter = image[-5]
        if letter == 'b':
            z_next = z_next + 0.33
        elif letter == 'c':
            z_next = z_next + 0.66

        x_corr = math.tan(angle_x) * z_next
        y_corr = math.tan(angle_y) * z_next

        z_image = math.sqrt(x_corr*x_corr + y_corr*y_corr)

        layer = math.sqrt(z_image*z_image + z_next*z_next)

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
                    (actual_point[0] - point[0]) ** 2) +
                    ((actual_point[1] - point[1]) ** 2))
            index = distance.index(min(distance))
            actual_point = points_whole_arm.pop(index)
            points_whole_arm_seg.append(actual_point)

        points_whole_arm_seg.append(first_point)

        file1 = open('model3D/' + model_name + '.txt', 'a')

        for point in range(0, len(points_whole_arm_seg)):
            str1 = str((points_whole_arm_seg[point][0]) * 0.33) + ';'\
                + str((points_whole_arm_seg[point][1]) * 0.33) + ';' \
                + str(layer) + '\n'
            file1.write(str1)
        file1.write("\n")
        file1.close()
