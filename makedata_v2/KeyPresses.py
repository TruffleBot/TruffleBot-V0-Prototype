from win32api import GetKeyState
import time
from ctypes import windll, Structure, c_long, byref
import os

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]



def queryMousePosition():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return { "x": pt.x, "y": pt.y}



VK_CODE = {
    'w':0x57,
    'a':0x41,
    's':0x53,
    'd':0x44,
    'tab':0x09,
    'spacebar':0x20,
    '1':0x31,
    '2':0x32,}

#array=[keystateW, keystateA, keystateS, keystateD, keystateTab, keystateSpace, keystate1, keystate2]

def isPressed(keycode):
    keystate = GetKeyState(keycode)
    if (keystate == 0) or (keystate == 1):
        return 0
    else:
        return 1
    

def keyPress():
    return [isPressed(0x57), isPressed(0x41), isPressed(0x53), isPressed(0x44),  isPressed(0x09), isPressed(0x20), isPressed(0x31), isPressed(0x32), isPressed(0x01), isPressed(0x02)]


def getOffset(old_pos, new_pos):
    return new_pos['x'] - old_pos['x'], new_pos['y'] - old_pos['y']

if __name__ == "__main__":
    last_position = queryMousePosition()
    while(True):
        # os.system('cls')
        # print(str(keyPress()))
        position = queryMousePosition()
        print(str(position))
        last_position = position