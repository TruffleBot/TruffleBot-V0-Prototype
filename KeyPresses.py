from win32api import GetKeyState
import time

VK_CODE = {
    'w':0x57,
    'a':0x41,
    's':0x53,
    'd':0x44,
    '1':0x31,
    'tab':0x09,
    'spacebar':0x20}

keystateW=0
keystateA=0
keystateS=0
keystateD=0
keystate1=0
keystateTab=0
keystateSpace=0

array=[keystateW, keystateA, keystateS, keystateD, keystate1, keystateTab, keystateSpace]

def isPressed(keycode):
    keystate = GetKeyState(keycode)
    if (keystate == 0) or (keystate == 1):
        return 0
    else:
        return 1
    

def keyPress():
    return [isPressed(0x57), isPressed(0x41), isPressed(0x53), isPressed(0x44), isPressed(0x31), isPressed(0x09), isPressed(0x20)]


if __name__ == "__main__":
    while(True):
        print(str(keyPress()))