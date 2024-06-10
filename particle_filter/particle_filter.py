import cv2 as cv
import numpy as np
import scipy
import scipy.ndimage
import params

class ParticleFilter:
  def __init__(self, width, height, particle_count):
    self.width = width
    self.height = height

    self.P = np.empty((params.N_particles, params.N_statedims + 1))
    self.P[:, 0] = np.random.randint(0, self.width, size=params.N_particles)
    self.P[:, 1] = np.random.randint(0, self.height, size=params.N_particles)
    self.P[:, 2] = np.random.randint(-10, 10, size=params.N_particles)
    self.P[:, 3] = np.random.randint(-10, 10, size=params.N_particles)
    self.P[:, 4] = 1.0/params.N_particles
  
  def ingest_image_mask(self, image_mask: cv.typing.MatLike):
    self.mask_count = cv.countNonZero(image_mask)
    self.mask = image_mask

    measurements = np.argwhere(image_mask == 255)
    measurements[:, [0, 1]] = measurements[:, [1, 0]] # swap the columns so each entry is (x, y) (not (y, x))

    self.measurements = measurements
  
  def calculate_closest_measurements(self):
    """
    Sets self.closest_measurements shape(len(P), 2): coordinates of the closest measurement for each particle
    Sets self.distances (currently not needed)
    """
    # where mask is zero we want a 1 and where mask is != 0 (255) we want a 0
    # edt distance transform calculates the distance from each 1 to the nearest 0
    # by passing `return_indices=True` we not only get the distance to the nearest measurement
    # but also the index of the nearest measurement for each pixel (the index is the position in the image)
    distances, indices = scipy.ndimage.distance_transform_edt(self.mask == 0, return_indices=True)

    # indices is of shape (2, height, width)
    y_indices, x_indices = indices
    x_pos_int = self.P[:, 0].astype('int')
    y_pos_int = self.P[:, 1].astype('int')
    closest_measurements_y = y_indices[y_pos_int, x_pos_int]
    closest_measurements_x = x_indices[y_pos_int, x_pos_int]
    self.closest_measurements = np.array([closest_measurements_x, closest_measurements_y]).T
    self.distances = distances[y_pos_int, x_pos_int] # we currently do not need this
  
  def measurement_update(self):
    # update/correction
    directions = self.closest_measurements - self.P[:, :2]

    # update position
    self.P[:, :2] += params.PARTICLE_FOLLOW_MEASUREMENT_SPEED * directions

    # update velocity
    self.P[:, 2:4] = self.P[:, 2:4]*0.9 + directions*0.1

  def prediction_update(self):
    # x = x + vx
    # y = y + vy
    self.P[:, :2] += self.P[:, 2:4]

  def add_noise(self, std: float):
    """
    Adds normal noise with standard deviation `std` to the position of the particles
    """
    self.P[:, :2] += np.random.normal(scale=std, size=(len(self.P), 2))

  def reseed_particles_to_measurements(self, fraction: float):
    """
    Takes a fraction of particles and sets their position onto a random measurement
    """
    # redistribute some particles to measurements
    number_to_redistribute = int(len(self.P) * params.PARTICLE_REDISTRIBUTION_FRACTION)
    # create random indices for the particles that get overwritten and for the measurements
    # we use to override the particles. One particle can only be overwritten once but a measurement
    # can be taken multiple times
    rand_particle_ind = np.random.choice(len(self.P), size=number_to_redistribute, replace=False)
    rand_meas_ind = np.random.choice(len(self.measurements), size=number_to_redistribute, replace=True)
    self.P[rand_particle_ind, :2] = self.measurements[rand_meas_ind]

  def clip_particles(self):
    # clip particles so they do not leave the screen
    self.P[:, :2] = self.P[:, :2].clip((0, 0), (self.width-1, self.height-1))

  def visualize_particles(self, image: cv.Mat):
    # visualize particles as little circles
    for x, y, vx, vy, _ in self.P:
      cv.circle(image, (int(x), int(y)), params.VIS_PARTICLE_RADIUS, params.VIS_PARTICLE_COLOR, 1)
      cv.line(image, (int(x), int(y)), (int(x+vx), int(y+vy)), params.VIS_PARTICLE_COLOR, 1)