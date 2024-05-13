import cv2 as cv
import numpy as np

class ColorpickerWindow:
  def __init__(self, name, initial_color=(0,0,0), initial_pos=(0, 0)) -> None:
    self.name = name
    self.img = np.zeros((300,512,3), np.uint8)

    self.h = initial_color[0]
    self.s = initial_color[1]
    self.v = initial_color[2]

    cv.namedWindow(self.name, cv.WINDOW_NORMAL)
    cv.moveWindow(self.name, initial_pos[0], initial_pos[1])

    cv.createTrackbar('H', self.name, self.h, 255, ColorpickerWindow.nothing)
    cv.createTrackbar('S', self.name, self.s, 255, ColorpickerWindow.nothing)
    cv.createTrackbar('V', self.name, self.v, 255, ColorpickerWindow.nothing)
  
  def update(self):
    self.h = cv.getTrackbarPos('H', self.name)
    self.s = cv.getTrackbarPos('S', self.name)
    self.v = cv.getTrackbarPos('V', self.name)

    hsv_color = np.uint8([[[self.h, self.s, self.v]]])
    bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)
    b, g, r = bgr_color[0][0]
    self.img[:] = [b, g, r]

    cv.imshow(self.name, self.img)
  
  def get_hsv(self):
    return np.array([self.h, self.s, self.v])
  
  def set_to_bgr(self, bgr, offset=0):
    bgr_color = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
    hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)
    hsv = hsv_color[0][0].astype('int16')

    hsv += np.uint16(offset)
    hsv = np.clip(hsv, 0, 255)
    hsv = hsv.astype('uint8')

    self.h, self.s, self.v = hsv

    cv.setTrackbarPos('H', self.name, self.h)
    cv.setTrackbarPos('S', self.name, self.s)
    cv.setTrackbarPos('V', self.name, self.v)

  def nothing(x):
    pass
 