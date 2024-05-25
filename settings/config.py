import cv2
import pyautogui as pag
import TwoHandTrack as tht

# detector initialization
detector = tht.HandDetector(detectionCon=0.7, maxHands=2)

# camera settings
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
wCam = 640
hCam = 480
cam.set(3, wCam)
cam.set(4, hCam)

# screen size
frameR = 150  # reducing the input window
wScr, hScr = pag.size()
