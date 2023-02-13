import cv2
import mediapipe as mp
import pyautogui as pag
import mouse
import numpy as np
import time
import TwoHandTrack as tht
import CVfunc as func
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import keyboard

# Camera ###########
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
wCam, hCam = 640, 480
cam.set(3, wCam)
cam.set(4, hCam)
####################
7
# FPS ##############
pTime = 0
####################

# Detector #########
detector = tht.HandDetector(detectionCon=0.7, maxHands=2)
####################

# Screen size ######
wScr, hScr = pag.size()
# print(wScr, hScr) # screen size output
frameR = 150  # reducing the input window
####################

# Smoothing the mouse
smooth = 3.5
plocX, plocY = 0, 0
clockX, clockY = 0, 0
####################

# Volume ###########

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
print(volRange)
minVol = volRange[0]
maxVol = volRange[1]
####################

# experemental distance
# x = [270, 220, 184, 159, 137, 121, 111, 98, 91, 82, 78, 71, 70, 44, 61, 57, 54, 52, 50]
# y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110]
#####################

# Flag ##############

flag = False
flag4time = False
#####################

prevLenght = 0
# Start
while (True):
    success, img = cam.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)
    # print(len(hands)) # Output numbers of hands

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] # List of 21 landmarks point
        bbox1 = hand1["bbox"] # Bounding box info x,y,w,h
        centerPoint1 = hand1["center"] # Center if the hand cx,cy
        handType1 = hand1["type"] # Hand type (left\right)
        fingers1 = detector.fingersUp(hand1)

        # angle of inclination + distance
        if len(lmList1) != 0:
            coord17x, coord17y = lmList1[17][1:]
            coord0x, coord0y = lmList1[0][1:]
            coord5x, coord5y = lmList1[5][1:]
            coord517x, coord517y = (coord17x + coord5x) / 2, (coord17y + coord5y) / 2
            shx17 = coord17x - coord0x
            shy17 = coord17y - coord0y
            shx517 = coord517x - coord0x
            shy517 = coord517y - coord0y
            ratioalpha = np.arctan(0)
            try:
                alphaplusbeta = np.arctan(shx517 / shy517)
            except ZeroDivisionError:
                alphaplusbeta = np.arctan(shx517 / (shy517 + 0.1))
                # alphaplusbeta = 1.57
            ratiobeta = -(alphaplusbeta - ratioalpha * 0)
            shxnew = (shx17 * np.cos(ratiobeta)) + (shy17 * np.sin(ratiobeta))
            shynew = (-shx17 * np.sin(ratiobeta)) + (shy17 * np.cos(ratiobeta))
            ratioXY = abs(shxnew / shynew)
            constratioXY = abs(-0.4)

            if ratioXY >= constratioXY:
                l = np.abs(shxnew * np.sqrt(1 + (1 / constratioXY) ** 2))
                distanse170cm = 5503.9283512 * l ** (-1.0016171)
            else:
                l = np.abs(shynew * np.sqrt(1 + constratioXY ** 2))
                distanse170cm = 5503.9283512 * l ** (-1.0016171)
            cv2.putText(img, f'{str(int(distanse170cm))}cm', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # We track the position of the index
        if len(lmList1) != 0:

            x1, y1 = lmList1[8][:2]

        # Check whether the finger is raised
        finup = detector.fingersUp(hand1)


        # frame restriction of hand movement
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        # Mod mouse movement
        if finup[0] == 0 and finup[1] == 1 and finup[2] == 0 and finup[3] == 0 and finup[4] == 0:
            # Coordinate conversion for the screen
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothing the mouse
            clockX = clockX + (x3 - plocX) / smooth
            clockY = clockY + (y3 - plocY) / smooth

            # Mouse movement
            mouse.move(clockX, clockY)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            plocX, plocY = clockX, clockY

        # Left mouse button
        if finup[0] == 0 and finup[1] == 1 and finup[2] == 1 and finup[3] == 0 and finup[4] == 0:
            length, info, img = detector.findDistance(lmList1[8][:2],lmList1[12][:2], img)
            print(length)
            # Mouse click if the distance is less than 25
            if length > (40):
                flag = True
            if length < (35) and flag == True:
                func.LCM(img, x1, y1, length)
                flag = False
        # Right mouse button
        if finup[0] == 0 and finup[1] == 1 and finup[2] == 0 and finup[3] == 0 and finup[4] == 1:
            length, info, img = detector.findDistance(lmList1[8][:2],lmList1[20][:2], img)
            # print(length)

            # Mouse click if the distance is less than 50
            if length > 30:
                flag = True
            if length < 50:
                func.RCM(img, x1, y1, length)
                flag = False

        # Grub and drop
        if finup[0] == 1 and finup[1] == 1 and finup[2] == 1 and finup[3] == 0 and finup[4] == 0:
            length, info, img = detector.findDistance(lmList1[8][:2],lmList1[12][:2], img)
            print(length)
            if length < 25:
                mouse.press(button="left")
                mouse.move(clockX, clockY)

        # scroll
        if finup[0] == 1 and finup[1] == 0 and finup[2] == 0 and finup[3] == 0 and finup[4] == 0:
            if len(lmList1) != 0:
                x1, y1 = lmList1[4][:2]
                x2, y2 = lmList1[5][:2]
                if y1 > y2:
                    mouse.wheel(delta=-0.5)
                elif y1 < y2:
                    mouse.wheel(delta=0.5)

    if len(hands) == 2:
        hand2 = hands[1]
        lmList2 = hand2["lmList"]  # List of 21 landmarks point
        bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
        centerPoint2 = hand2["center"]  # Center if the hand cx,cy
        handType2 = hand2["type"]  # Hand type (left\right)
        fingers2 = detector.fingersUp(hand2)
        #print(fingers1,fingers2)
        #length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)


        if finup[0] == 1 and finup[1] == 1 and finup[2] == 0 and finup[3] == 0 and finup[4] == 0 and fingers2[0] == 1 and fingers2[1] == 1 and fingers2[2] == 0 and fingers2[3] == 0 and fingers2[4] == 0:
            length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)
            print(length, prevLenght)
            if (length > prevLenght):
                print(length, prevLenght)
                keyboard.press('ctrl')
                print("fsef")
                mouse.wheel(0.1)
                print("adadfd")
                keyboard.release('ctrl')
                # nextlenght = lenght
                prevLenght = length
            else:
                print("abobobob")
                keyboard.press('ctrl')
                mouse.wheel(-0.1)
                keyboard.release('ctrl')
                prevLenght = length
            print(length, prevLenght)


    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display
    cv2.imshow("Hand tracking", img)
    cv2.waitKey(1)

