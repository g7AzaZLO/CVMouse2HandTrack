import cv2
import mouse
import numpy as np
import time
import asyncio

import pyautogui

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


def handle_screenshot(finup: List[int], start_time: float, img: Any) -> float:
    """Handles taking a screenshot if a fist is shown for 3 seconds."""
    if finup == [0, 0, 0, 0, 0]:  # All fingers down
        if start_time == 0:
            start_time = time.time()
        elapsed_time = time.time() - start_time
        if elapsed_time >= 3:
            pyautogui.screenshot('screenshot.png')
            start_time = 0  # Reset start time
        else:
            cv2.putText(img, f'{3 - int(elapsed_time)}', (wCam // 2, hCam // 2), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255),
                        5)
    else:
        start_time = 0  # Reset start time if gesture is not maintained
    return start_time


def detect_hands(img):
    """Wrapper function to call detector.findHands with keyword arguments."""
    return detector.findHands(img, flipType=False)


def handle_arrow_keys(hands: List[Any], start_time_up: float, start_time_down: float, arrow_keys_active: bool, direction_triggered: str, action_performed: bool, img: Any) -> Tuple[float, float, bool, str, bool]:
    """Handles pressing arrow keys based on thumb gestures for 3 seconds."""
    current_time = time.time()
    thumbs_up_detected = False
    thumbs_down_detected = False

    if len(hands) == 2:
        hand1, hand2 = hands
        fingers1 = detector.fingersUp(hand1)
        fingers2 = detector.fingersUp(hand2)

        # Thumbs up
        if fingers1 == [1, 0, 0, 0, 0] and fingers2 == [1, 0, 0, 0, 0]:
            if not arrow_keys_active:
                start_time_up = current_time
                arrow_keys_active = True
                direction_triggered = "right"
                action_performed = False
            elapsed_time = current_time - start_time_up
            cv2.putText(img, f'{3 - int(elapsed_time)}', (wCam//2, hCam//2 + 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
            if elapsed_time >= 3 and not action_performed:
                keyboard.press_and_release('right')
                action_performed = True
            thumbs_up_detected = True

        # Thumbs down
        elif fingers1 == [0, 0, 0, 0, 1] and fingers2 == [0, 0, 0, 0, 1]:
            if not arrow_keys_active:
                start_time_down = current_time
                arrow_keys_active = True
                direction_triggered = "left"
                action_performed = False
            elapsed_time = current_time - start_time_down
            cv2.putText(img, f'{3 - int(elapsed_time)}', (wCam//2, hCam//2 + 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
            if elapsed_time >= 3 and not action_performed:
                keyboard.press_and_release('left')
                action_performed = True
            thumbs_down_detected = True

        # Reset if no continuous gesture is detected
        if not thumbs_up_detected and not thumbs_down_detected:
            arrow_keys_active = False
            start_time_up = 0
            start_time_down = 0
            direction_triggered = ""
            action_performed = False
    else:
        arrow_keys_active = False
        start_time_up = 0
        start_time_down = 0
        direction_triggered = ""
        action_performed = False

    return start_time_up, start_time_down, arrow_keys_active, direction_triggered, action_performed


def handle_window_switch(hands: List[Any], window_switch_start_pos: Tuple[int, int], window_switch_active: bool, window_switch_direction: str, last_switch_time: float, img: Any) -> Tuple[Tuple[int, int], bool, str, float]:
    """Handles switching between open windows based on hand gestures."""
    current_time = time.time()
    cooldown_period = 1  # Cooldown period in seconds

    if len(hands) == 1:
        hand1 = hands[0]
        fingers1 = detector.fingersUp(hand1)
        lmList1 = hand1["lmList"]

        # Check for the "W" gesture
        if fingers1 == [0, 1, 1, 1, 0]:
            x1, y1 = lmList1[8][:2]
            if not window_switch_active:
                # Record the initial position
                window_switch_start_pos = (x1, y1)
                window_switch_active = True
                window_switch_direction = ""
                cv2.putText(img, 'Start', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                # Determine direction of movement
                if window_switch_direction == "":
                    if x1 > window_switch_start_pos[0] + 100:  # Adjust sensitivity as needed
                        window_switch_direction = "right"
                    elif x1 < window_switch_start_pos[0] - 100:  # Adjust sensitivity as needed
                        window_switch_direction = "left"

                if window_switch_direction == "right" and x1 > window_switch_start_pos[0] + 100 and current_time - last_switch_time > cooldown_period:
                    keyboard.press_and_release('alt+tab')
                    window_switch_active = False
                    last_switch_time = current_time
                elif window_switch_direction == "left" and x1 < window_switch_start_pos[0] - 100 and current_time - last_switch_time > cooldown_period:
                    keyboard.press_and_release('alt+shift+tab')
                    window_switch_active = False
                    last_switch_time = current_time
        else:
            window_switch_active = False
            window_switch_start_pos = (0, 0)
            window_switch_direction = ""
    else:
        window_switch_active = False
        window_switch_start_pos = (0, 0)
        window_switch_direction = ""

    return window_switch_start_pos, window_switch_active, window_switch_direction, last_switch_time

import os
import time

def handle_task_manager(hands: List[Any], task_manager_start_time: float, task_manager_active: bool, img: Any) -> Tuple[float, bool]:
    """Handles opening the Task Manager based on hand gestures."""
    current_time = time.time()

    if len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]
        fingers1 = detector.fingersUp(hand1)
        fingers2 = detector.fingersUp(hand2)

        # Check for both hands making a fist
        if fingers1 == [0, 0, 0, 0, 0] and fingers2 == [0, 0, 0, 0, 0]:
            if not task_manager_active:
                if task_manager_start_time == 0:
                    task_manager_start_time = current_time
                elapsed_time = current_time - task_manager_start_time
                cv2.putText(img, f'Opening Task Manager in {int(3 - elapsed_time)}', (wCam//2 - 100, hCam//2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if elapsed_time >= 3:
                    os.system("taskmgr")
                    task_manager_active = True
                    task_manager_start_time = 0  # Reset the timer
        else:
            task_manager_start_time = 0
            task_manager_active = False
    else:
        task_manager_start_time = 0
        task_manager_active = False

    return task_manager_start_time, task_manager_active

import subprocess

import subprocess

def handle_notepad(hands: List[Any], notepad_start_time: float, notepad_active: bool, img: Any) -> Tuple[float, bool]:
    """Handles opening Notepad and typing a message based on hand gestures."""
    current_time = time.time()

    if len(hands) == 1:
        hand1 = hands[0]
        fingers1 = detector.fingersUp(hand1)

        # Check for middle finger up (assuming [0, 0, 1, 0, 0] means only middle finger is up)
        if fingers1 == [0, 0, 1, 0, 0]:
            if not notepad_active:
                if notepad_start_time == 0:
                    notepad_start_time = current_time
                elapsed_time = current_time - notepad_start_time
                cv2.putText(img, f'Opening Notepad in {int(3 - elapsed_time)}', (wCam // 2 - 100, hCam // 2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if elapsed_time >= 3:
                    subprocess.Popen(['notepad.exe'])
                    time.sleep(1)  # Wait for Notepad to open
                    keyboard.write('Сейчас в ответку прилетит')
                    notepad_active = True
                    notepad_start_time = 0  # Reset the timer
        else:
            notepad_start_time = 0
            notepad_active = False
    else:
        notepad_start_time = 0
        notepad_active = False

    return notepad_start_time, notepad_active

async def process_frame(executor):
    """Asynchronously processes video frames and gesture handling."""
    minVol, maxVol, volume = get_volume_range()
    global plocX, plocY, clockX, clockY, pTime, flag, prevLength

    volume_control_active = False
    brightness_control_active = False
    task_manager_active = False
    notepad_active = False
    distance_buffer = deque()

    screenshot_start_time = 0
    thumbs_up_start_time = 0
    thumbs_down_start_time = 0
    arrow_keys_active = False
    direction_triggered = ""
    action_performed = False

    window_switch_start_time = 0
    window_switch_start_pos = (0, 0)
    window_switch_active = False
    window_switch_direction = ""
    last_switch_time = 0
    task_manager_start_time = 0
    notepad_start_time = 0

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

            if not arrow_keys_active and not window_switch_active:
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

                screenshot_start_time = handle_screenshot(finup, screenshot_start_time, img)

                if len(hands) == 2:
                    hand2 = hands[1]
                    lmList2 = hand2["lmList"]
                    centerPoint1 = hand1["center"]
                    centerPoint2 = hand2["center"]
                    fingers2 = detector.fingersUp(hand2)

                    prevLength = handle_zoom(finup, fingers2, centerPoint1, centerPoint2, prevLength, img)

            thumbs_up_start_time, thumbs_down_start_time, arrow_keys_active, direction_triggered, action_performed = handle_arrow_keys(
                hands, thumbs_up_start_time, thumbs_down_start_time, arrow_keys_active, direction_triggered, action_performed, img
            )

            window_switch_start_pos, window_switch_active, window_switch_direction, last_switch_time = handle_window_switch(
                hands, window_switch_start_pos, window_switch_active, window_switch_direction, last_switch_time, img
            )

            task_manager_start_time, task_manager_active = handle_task_manager(
                hands, task_manager_start_time, task_manager_active, img
            )

            notepad_start_time, notepad_active = handle_notepad(
                hands, notepad_start_time, notepad_active, img
            )

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
