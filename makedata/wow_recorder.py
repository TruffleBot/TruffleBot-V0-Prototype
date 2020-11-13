import win32gui
import cv2
from grab_screen import grab_screen
from KeyPresses import keyPress

def get_wow_handle():
    return win32gui.FindWindow('GxWindowClass', 'World of Warcraft')

def get_recording_region(handle, show=False):
        rect = win32gui.GetWindowRect(handle) 
        rect = list(rect)
        rect[0] = rect[0] + 8
        rect[1] = rect[1] + 31
        rect[2] = rect[2] - 9
        rect[3] = rect[3] - 9
        x = rect[0]
        y = rect[1]
        w = rect[2] - x
        h = rect[3] - y

        if show:
            print(rect, w, h)
            printscreen = grab_screen(rect)

            cv2.imshow('Recording Region', printscreen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        
        return rect

class WoWRecorder:
    
    def __init__(self):
        self.handle = get_wow_handle()
 
    def get_shape(self):
        rect = get_recording_region(self.handle)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        return (w, h)

    def get_frame(self):
        return grab_screen(get_recording_region(self.handle))
    
    # def get_wow_running(self):
    #     try:
    #         rect = win32gui.GetWindowRect(self.handle)
    #         return True
    #     except:
    #         return False

    def get_keys(self):
        return keyPress()
    
