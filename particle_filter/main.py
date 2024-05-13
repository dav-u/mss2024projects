import cv2 as cv
import numpy as np
from colorpicker_window import ColorpickerWindow

# create windows and move them next to each other
MASK_WINDOW = 'Mask'
WEBCAM_WINDOW = 'Webcam'
cv.namedWindow(MASK_WINDOW, cv.WINDOW_NORMAL)
cv.namedWindow(WEBCAM_WINDOW, cv.WINDOW_NORMAL)

cv.moveWindow(MASK_WINDOW, 0, 0)
cv.moveWindow(WEBCAM_WINDOW, 600, 0)
cv.resizeWindow(MASK_WINDOW, 600, 600)
cv.resizeWindow(WEBCAM_WINDOW, 600, 600)

lower_bound_picker = ColorpickerWindow('Lower bound color picker', initial_color=(0, 100, 70), initial_pos=(1200, 0))
upper_bound_picker = ColorpickerWindow('Upper bound color picker', initial_color=(30, 255, 255), initial_pos=(1200, 400))

image: cv.Mat = None

def webcam_click(event, x, y, flags, param):
  if event != cv.EVENT_LBUTTONDOWN:
    return

  color = image[y, x]
  lower_bound_picker.set_to_bgr(color, offset=-20)
  upper_bound_picker.set_to_bgr(color, offset=+20)

cv.setMouseCallback(WEBCAM_WINDOW, webcam_click)

capture = cv.VideoCapture(0)

# Define lower and upper bounds for orange in HSV
lower_orange = np.array([0, 100, 70])
upper_orange = np.array([30, 255, 255])

if not capture.isOpened():
  print("Could not open camera")
  exit()

while True:
  upper_bound_picker.update()
  lower_bound_picker.update()
  lower_color_bound = lower_bound_picker.get_hsv()
  upper_color_bound = upper_bound_picker.get_hsv()

  _, image = capture.read()

  hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
  mask = cv.inRange(hsv_image, lower_color_bound, upper_color_bound)
  contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

  cv.drawContours(image, contours, -1, (0,255,0), 3)

  measurements = []
  for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    # weight = (w*h) / (image.shape[0] * image.shape[1])
    weight = cv.contourArea(contour) / (image.shape[0] * image.shape[1])
    measurements.append((x+w//2, y+h//2, weight))
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  
  # (x coordinate, y coordinate, weight)
  for (x, y, w) in measurements:
    cv.circle(image, (x, y), int(500 * w), (0, 255, 255), -1)
  
  cv.imshow(MASK_WINDOW, mask)
  cv.imshow(WEBCAM_WINDOW, image)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv.destroyAllWindows()