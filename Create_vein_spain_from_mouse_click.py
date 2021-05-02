import cv2
from os.path import isfile, join
from os import listdir

# the [x, y] for each right-click event will be stored here
right_clicks = list()

# this function will be called whenever the mouse is right-clicked


def mouse_callback(event, x, y, flags, params):

    # right-click event value is 2
    if event == 1:
        global right_clicks

        # store the coordinates of the right-click event
        right_clicks.append([img_name, x, y])

        # this just verifies that the mouse data is being collected
        # you probably want to remove this later
        print(right_clicks)


mypath = 'results/0_color'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()

for img_name in onlyfiles:
    img = cv2.imread(mypath + '/' + img_name, cv2.IMREAD_COLOR)
    scale_width = 640 / img.shape[1]
    scale_height = 480 / img.shape[0]
    scale = min(scale_width, scale_height)*2
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)

    # set mouse callback function for window
    cv2.setMouseCallback('image', mouse_callback)

    cv2.imshow('image', img)
    cv2.waitKey(0)

with open('your_file.txt', 'w') as f:
    for item in right_clicks:
        f.write("%s\n" % item)
