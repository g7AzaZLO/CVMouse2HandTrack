import cv2
import mouse
import numpy as np
import time
import asyncio
from functional import CVfunc as func
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from collections import deque
import keyboard
from settings.config import detector, cam, wCam, hCam, frameR, wScr, hScr
from settings.initial_var import smooth, plocX, plocY, clockX, clockY, pTime, flag, prevLength
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Any


def get_volume_range() -> Tuple[float, float, Any]:
    """Gets the volume range from the system."""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    return volRange[0], volRange[1], volume


def compute_distance(lmList1: List[List[int]]) -> float:
    """Computes the distance between landmarks for depth estimation."""
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
    ratiobeta = -(alphaplusbeta - ratioalpha * 0)
    shxnew = (shx17 * np.cos(ratiobeta)) + (shy17 * np.sin(ratiobeta))
    shynew = (-shx17 * np.sin(ratiobeta)) + (shy17 * np.cos(ratiobeta))
    ratioXY = abs(shxnew / shynew)
    constratioXY = abs(-0.4)

    if ratioXY >= constratioXY:
        l = np.abs(shxnew * np.sqrt(1 + (1 / constratioXY) ** 2))
        distance170cm = 5503.9283512 * l ** (-1.0016171)
    else:
        l = np.abs(shynew * np.sqrt(1 + constratioXY ** 2))
        distance170cm = 5503.9283512 * l ** (-1.0016171)
    return distance170cm


def move_mouse(x1: int, y1: int, plocX: float, plocY: float, clockX: float, clockY: float, smooth: int) -> Tuple[
    float, float]:
    """Moves the mouse cursor smoothly based on hand landmarks."""
    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
    clockX = clockX + (x3 - plocX) / smooth
    clockY = clockY + (y3 - plocY) / smooth
    mouse.move(clockX, clockY)
    return clockX, clockY


def handle_mouse_click(finup: List[int], lmList1: List[List[int]], img: Any, x1: int, y1: int) -> bool:
    """Handles left and right mouse clicks based on finger positions."""
    global flag
    if finup[0] == 0 and finup[1] == 1 and finup[2] == 1 and finup[3] == 0 and finup[4] == 0:
        length, info, img = detector.findDistance(lmList1[8][:2], lmList1[12][:2], img)
        if length > 40:
            flag = True
        if length < 35 and flag:
            func.LCM(img, x1, y1, length)
            flag = False
    if finup[0] == 0 and finup[1] == 1 and finup[2] == 0 and finup[3] == 0 and finup[4] == 1:
        length, info, img = detector.findDistance(lmList1[8][:2], lmList1[20][:2], img)
        if length > 30:
            flag = True
        if length < 50:
            func.RCM(img, x1, y1, length)
            flag = False
    return flag


def handle_scroll(finup: List[int], lmList1: List[List[int]]):
    """Handles mouse scrolling based on finger positions."""
    if finup[0] == 1 and finup[1] == 0 and finup[2] == 0 and finup[3] == 0 and finup[4] == 0:
        x1, y1 = lmList1[4][:2]
        x2, y2 = lmList1[5][:2]
        if y1 > y2:
            mouse.wheel(delta=-0.5)
        elif y1 < y2:
            mouse.wheel(delta=0.5)


def handle_zoom(finup: List[int], fingers2: List[int], centerPoint1: Tuple[int, int], centerPoint2: Tuple[int, int],
                prevLength: float, img: Any) -> float:
    """Handles zooming in and out based on the distance between two hands."""
    global flag
    if finup[0] == 1 and finup[1] == 1 and finup[2] == 0 and finup[3] == 0 and finup[4] == 0 and fingers2[0] == 1 and \
            fingers2[1] == 1 and fingers2[2] == 0 and fingers2[3] == 0 and fingers2[4] == 0:
        length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)
        if length > prevLength:
            keyboard.press('ctrl')
            mouse.wheel(0.1)
            keyboard.release('ctrl')
        else:
            keyboard.press('ctrl')
            mouse.wheel(-0.1)
            keyboard.release('ctrl')
        prevLength = length
    return prevLength


def handle_grab_and_drop(finup: List[int], lmList1: List[List[int]], clockX: float, clockY: float, img: Any):
    """Handles grabbing and moving objects based on finger positions."""
    if finup[0] == 1 and finup[1] == 1 and finup[2] == 1 and finup[3] == 0 and finup[4] == 0:
        length, info, img = detector.findDistance(lmList1[8][:2], lmList1[12][:2], img)
        print(length)
        if length < 25:
            mouse.press(button="left")
            mouse.move(clockX, clockY)


def handle_volume_control(finup: List[int], lmList1: List[List[int]], minVol: float, maxVol: float, volume: Any,
                          img: Any, distance_buffer: deque, buffer_size: int = 5) -> bool:
    """Handles volume control based on finger positions using moving average."""
    if finup[0] == 1 and finup[1] == 1 and finup[2] == 0 and finup[3] == 0 and finup[4] == 1:
        length, info, img = detector.findDistance(lmList1[4][:2], lmList1[8][:2], img)
        dist = compute_distance(lmList1)

        # Update the distance buffer
        distance_buffer.append(length)
        if len(distance_buffer) > buffer_size:
            distance_buffer.popleft()

        # Compute the moving average
        smoothed_length = np.mean(distance_buffer)

        func.chngVol(smoothed_length, minVol, maxVol, volume, dist)
        return True
    return False


def detect_hands(img):
    """Wrapper function to call detector.findHands with keyword arguments."""
    return detector.findHands(img, flipType=False)


async def process_frame(executor):
    """Asynchronously processes video frames and gesture handling."""
    minVol, maxVol, volume = get_volume_range()
    global plocX, plocY, clockX, clockY, pTime, flag, prevLength

    volume_control_active = False
    distance_buffer = deque()

    while True:
        success, img = await asyncio.get_event_loop().run_in_executor(executor, cam.read)
        if not success:
            break
        img = cv2.flip(img, 1)  # Flip the image
        result = await asyncio.get_event_loop().run_in_executor(executor, detect_hands, img)

        if result is None or len(result) == 0:
            hands = []
        elif isinstance(result, tuple) and len(result) == 2:
            hands, img = result
        else:
            hands = result if isinstance(result, list) else []
            img = result[1] if len(result) > 1 else img

        if hands and len(hands) > 0:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            fingers1 = detector.fingersUp(hand1)

            if len(lmList1) != 0:
                dist = compute_distance(lmList1)
                cv2.putText(img, f'{str(int(dist))}cm', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            if len(lmList1) != 0:
                x1, y1 = lmList1[8][:2]

            finup = detector.fingersUp(hand1)

            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            if finup[0] == 0 and finup[1] == 1 and finup[2] == 0 and finup[3] == 0 and finup[4] == 0:
                clockX, clockY = move_mouse(x1, y1, plocX, plocY, clockX, clockY, smooth)
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                plocX, plocY = clockX, clockY

            flag = handle_mouse_click(finup, lmList1, img, x1, y1)
            handle_scroll(finup, lmList1)
            handle_grab_and_drop(finup, lmList1, clockX, clockY, img)

            volume_control_active = handle_volume_control(finup, lmList1, minVol, maxVol, volume, img, distance_buffer)

            if volume_control_active and finup[4] == 0:
                volume_control_active = False

            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                centerPoint1 = hand1["center"]
                centerPoint2 = hand2["center"]
                fingers2 = detector.fingersUp(hand2)

                prevLength = handle_zoom(finup, fingers2, centerPoint1, centerPoint2, prevLength, img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Hand tracking", img)
        cv2.waitKey(1)
        await asyncio.sleep(0)  # Allows other asynchronous tasks to run

async def main():
    executor = ThreadPoolExecutor(max_workers=4)
    await process_frame(executor)

if __name__ == "__main__":
    asyncio.run(main())