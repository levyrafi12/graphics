import cv2
import argparse
import numpy as np
from pathlib import Path

# different pre defined point colors
point_colors = '''6929c4
1192e8
005d5d
9f1853
fa4d56
570408
198038
002d9c
ee538b
b28600
009d9a
012749
8a3800
a56eff'''.split('\n')


def hex2rgb(s: str):
    c = []
    for i in range(0, len(s), 2):
        c.append(float(int(s[i:i + 2], 16)))
    return np.array(c)


# convert the hex text colors to RGB
point_colors = np.stack([hex2rgb(s) for s in point_colors], axis=0)
num_colors = len(point_colors)

# list to save tag points
lst = []
counter = 0


def to_txt(lst):
    '''
    convert list to text format
    :param lst: list of tuples
    :return: the formatted string
    '''
    txt = ''
    for (x, y) in lst:
        txt += f'{x} {y}\n'
    return txt.strip()


def write_to_file(path, txt):
    with open(path, 'w+') as file:
        file.write(txt)


def draw_circle(event,x,y,flags,param):
    '''
    Callback on press
    :param event: event type
    :param x: x pixel position
    :param y: y pixel position
    :param flags:
    :param param:
    :return:
    '''
    global img
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw a circle at tag location
        cv2.circle(img, (x,y), 5, point_colors[counter % num_colors], -1)
        counter += 1
        # save the point to a list
        lst.append([x, y])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('tag corresponding points')
    parser.add_argument('--i', type=Path, required=True,
                        help='path to read image from')

    parser.add_argument('--o', type=Path, required=True,
                        help='path to save pt to')
    args = parser.parse_args()
    img = cv2.imread(str(args.i))
    winname = "Paint :: Press ESC to exit; Click to TAG"
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, draw_circle)

    k = 0
    while k != 27:
        # wait until ESC is pressed
        cv2.imshow(winname, img)
        k = cv2.waitKey(20) & 0xFF
    if len(lst) != 0:
        # if escape pressed save the tagged point list to a file
        cv2.destroyAllWindows()
        pts_txt = to_txt(lst)
        print(pts_txt)
        write_to_file(args.o, pts_txt)
    else:
        # if no tags were made
        print('no points')

