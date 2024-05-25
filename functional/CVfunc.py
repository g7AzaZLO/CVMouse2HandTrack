# Program by ZLO#DEV

import cv2
import mouse
import numpy as np



def LCM(img, x1, y1, length):
    cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
    mouse.click('left')


def RCM(img, x1, y1, length):
    cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
    mouse.click(button='right')


def chngVol(length, minVol, maxVol, volume, distanse170cm):
    coff = distanse170cm / 100
    length = length * coff
    # vol = np.interp(length, [10, 90], [-82, 0])
    vol = np.interp(length, [10, 90], [minVol, maxVol])
    print(length, vol)
    volume.SetMasterVolumeLevel(vol, None)


