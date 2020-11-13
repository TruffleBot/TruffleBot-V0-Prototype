import win32gui, win32api, win32con
import cv2
from grab_screen import grab_screen
from KeyPresses import keyPress, queryMousePosition, getOffset
import os
import time
import threading

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
        self.last_position = self.get_cursorpos()
        self.camera_start_positon = self.get_cursorpos()
 
    def get_shape(self):
        rect = get_recording_region(self.handle)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        return (w, h)

    def get_frame(self):
        return grab_screen(get_recording_region(self.handle))

    def get_cursorpos(self):
        return queryMousePosition()

    def set_size(self, w, h):
        rect = win32gui.GetWindowRect(self.handle) 
        rect = list(rect)
        x = rect[0]
        y = rect[1]
        win32gui.MoveWindow(self.handle, x, y, w+17, h+40, True)

    def show_recording_region(self):
        get_recording_region(self.handle, show=True)
    
    def offset_helper(self):
        position = self.get_cursorpos()
        offset = getOffset(self.last_position, position)
        return offset

    def camera_offset_helper(self):
        last_position = self.get_cursorpos()
        has_reset = True
        x = 0
        y = 0
        keys = self.get_keys()

        while(keys[-1] == 1 or keys[-2] == 1):
            position = self.get_cursorpos()

            if has_reset:
                dx, dy = getOffset(self.camera_start_positon, position)
                x += dx
                y += dy
                has_reset = False

            
            if position['x'] == self.camera_start_positon ['x'] and position['y'] == self.camera_start_positon ['y'] :
                has_reset = True
            else:
                dx, dy = getOffset(last_position, position)
                x += dx
                y += dy
                
            last_position = position
            keys = self.get_keys()
            
            # os.system('cls')
            # print(x,y)


            # loop_time = time.time()-last_time
            # # print('loop took {} seconds'.format(loop_time))
            # FPS = 100
            # if 1/FPS > loop_time:
            #     sleep_time = 1/FPS - loop_time
            #     # print('sleeping for {} seconds because we have extra time'.format(sleep_time))
            #     time.sleep(sleep_time)

            # last_time = time.time()

                
        return x, y


    def get_offset(self):
        
        keys = self.get_keys()
        res = None
        if(keys[-1] == 1 or keys[-2] == 1):
            self.calibrate_camera_start_pos()
            res = self.camera_offset_helper()
        else:
            res = self.offset_helper()

        self.last_position = self.get_cursorpos()

        return res

    def calibrate_camera_start_pos(self):
        rect = win32gui.GetWindowRect(self.handle) 
        rect = list(rect)
        x = rect[0]
        y = rect[1]
        w = (rect[2] - x)
        h = (rect[3] - y)
        self.camera_start_positon = {'x': x+w//2, 'y': y+h//2}
        return self.camera_start_positon
            



    
    # def get_wow_running(self):
    #     try:
    #         rect = win32gui.GetWindowRect(self.handle)
    #         return True
    #     except:
    #         return False

    def get_keys(self):
        return keyPress()
    

def mouse_mover(recorder):
    time.sleep(2)
    print('moving mouse')
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
    time.sleep(1)
    # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 100,100,0,0)
    for _ in range(50):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 1,0,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0,1,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0,1,0,0)
        # print(str(recorder.calibrate_camera_start_pos()))
        # print(str(queryMousePosition()))
        time.sleep(.01)
    time.sleep(2)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)

if __name__ == "__main__":
    scale = 3
    recorder = WoWRecorder()
    recorder.set_size(224*scale,224*scale)
    
    # mouse_t = threading.Thread(target=mouse_mover, args=(recorder,),daemon=True)
    # mouse_t.start()

    while(True):

        os.system('cls')
        
        #mouse calibration debug
        # print(str(queryMousePosition()))
        # print(str(recorder.calibrate_camera_start_pos()))
        
        print(str(recorder.get_keys())) 
        print(str(recorder.get_offset()))

        # recorder.show_recording_region()
        time.sleep(1)

