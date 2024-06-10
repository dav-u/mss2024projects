import cv2 as cv
import numpy as np
import scipy.ndimage
from colorpicker_window import ColorpickerWindow
import params
import scipy

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

P = np.empty((params.N_particles, params.N_statedims + 1))
P[:, 0] = np.random.randint(0, image_width, size=params.N_particles)
P[:, 1] = np.random.randint(0, image_height, size=params.N_particles)
P[:, 2] = np.random.randint(-10, 10, size=params.N_particles)
P[:, 3] = np.random.randint(-10, 10, size=params.N_particles)
P[:, 4] = 1.0/params.N_particles

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
  
  # visualize particles as little circles
  for x, y, vx, vy, _ in P:
    cv.circle(image, (int(x), int(y)), params.VIS_PARTICLE_RADIUS, params.VIS_PARTICLE_COLOR, -1)
    
  # where mask is zero we want a 1 and where mask is != 0 (255) we want a 0
  # edt distance transform calculates the distance from each 1 to the nearest 0
  # bei passing `return_indices=True` we not only get the distance to the nearest measurement
  # but also the index of the nearest measurement for each pixel
  distances, indices = scipy.ndimage.distance_transform_edt(mask == 0, return_indices=True)
  # indices is of shape (2, height, width)
  y_indices, x_indices = indices
  x_pos_int = P[:, 0].astype('int')
  y_pos_int = P[:, 1].astype('int')
  closest_measurements_y = y_indices[y_pos_int, x_pos_int]
  closest_measurements_x = x_indices[y_pos_int, x_pos_int]
  closest_measurements = np.array([closest_measurements_x, closest_measurements_y]).T
  distances = distances[y_pos_int, x_pos_int] # we currently do not need this

  # update/correction
  P[:, :2] += params.PARTICLE_FOLLOW_MEASUREMENT_SPEED * (closest_measurements - P[:, :2])

  # add noise
  P[:, :2] += np.random.normal(scale=params.PARTICLE_NOISE, size=(len(P), 2))

  # redistribute some particles uniformly
  number_to_redistribute = int(len(P) * params.PARTICLE_REDISTRIBUTION_FRACTION)
  rand_ind = np.random.choice(np.arange(len(P)), size=number_to_redistribute, replace=False)
  P[rand_ind, :2] = np.random.randint((0, 0), (image_width, image_height), size=(len(rand_ind), 2))

  # clip particles so they do not leave the screen
  P[:, :2] = P[:, :2].clip((0, 0), (image_width-1, image_height-1))

  cv.imshow(MASK_WINDOW, mask)
  cv.imshow(WEBCAM_WINDOW, image)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv.destroyAllWindows()