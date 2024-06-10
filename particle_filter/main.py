import cv2 as cv
import numpy as np
from colorpicker_window import ColorpickerWindow
import params
from particle_filter import ParticleFilter

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

original_image: cv.Mat = None

def webcam_click(event, x, y, flags, param):
  if event != cv.EVENT_LBUTTONDOWN:
    return

  color = original_image[y, x]
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

image_height, image_width, _ = capture.read()[1].shape


particle_filter = ParticleFilter(image_width, image_height, params.N_particles)

while True:
  upper_bound_picker.update()
  lower_bound_picker.update()
  lower_color_bound = lower_bound_picker.get_hsv()
  upper_color_bound = upper_bound_picker.get_hsv()

  _, original_image = capture.read()
  image = original_image.copy()

  hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
  mask = cv.inRange(hsv_image, lower_color_bound, upper_color_bound)
  mask_count = cv.countNonZero(mask)

  if mask_count == 0: continue

  particle_filter.ingest_image_mask(mask)
  particle_filter.visualize_particles(image)
  particle_filter.calculate_closest_measurements()
  particle_filter.measurement_update()
  particle_filter.add_noise(params.PARTICLE_NOISE)
  particle_filter.reseed_particles_to_measurements(params.PARTICLE_REDISTRIBUTION_FRACTION)
  particle_filter.clip_particles()

  cv.imshow(MASK_WINDOW, mask)
  cv.imshow(WEBCAM_WINDOW, image)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv.destroyAllWindows()