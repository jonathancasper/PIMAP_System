"""PIMAP Analyze component that performs Objective Mobility Analysis.

PIMAP Analyze Objective Mobility is a PIMAP Analyze component that analyzes
pressure bandage PIMAP samples and outputs PIMAP metrics that report the
amount and strength of movements. The following is a citation that describes
how the analysis works in detail:

S. Mansfield, S. Rangarajan, K. Obraczka, H. Lee, D. Young, and S. Roy,
“Objective Pressure Injury Risk Assessment Using A Wearable Pressure Sensor,”
in 2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM),
San Diego, CA, USA, Nov. 2019, pp. 1561–1568, doi: 10.1109/BIBM47256.2019.8982939.

Pressure bandage data is contained inside of a pressure_bandage PIMAP sample and
is contained inside a nested list based on the following:
  key: [x, y], c0: centroid 0, c1: centroid 1, c2: centroid 2

     c0          c0         c0         c0
  [[0, 0],   [1/3, 0],   [2/3, 0],   [1, 0],
     c1          c0         c0         c2
   [0, 1/3], [1/3, 1/3], [2/3, 1/3], [1, 1/3],
     c1          c1         c2         c2
   [0, 2/3], [1/3, 2/3], [2/3, 2/3], [1, 2/3],
     c1          c1         c2         c2
   [0, 1],   [1/3, 1],   [2/3, 1],   [1, 1]]

License:
Author: Sam Mansfield
"""
import ast
import numpy as np
import sys
import time
from collections import defaultdict
from pimap import pimaputilities as pu

class PimapAnalyzeObjectiveMobility:
  def __init__(self, max_pressure=100.0, sample_type="pressure_bandage",
               system_samples=False, app=""):
    """Constructor for PIMAP Analyze Objective Mobility

    Arguments:
      max_pressure (optional): A float value used to normalize pressure values. The units
        should match the pressure_bandage values. Defaults to 100.0
      sample_type (optional): A string value to filter by sample type of incoming data.
        Defaults to "pressure_bandage".
      system_samples (optional): A boolean value that indicates whether system_samples
        are produced that report the throughput and other relevant values of this
        component. Defaults to False.
      app (optional): A name of the application running, which is used to append
        to the name of they sample_type of system_samples,
        e.g. sample_type:"system_samples_app". Defaults to "".

    Exceptions:
      ValueError:
        If a non-float max_pressure is given.
    """
    self.max_pressure = float(max_pressure)
    self.sample_type = str(sample_type)
    self.system_samples = system_samples
    self.system_samples_updated = time.time()
    self.system_samples_update_period = 1.0
    self.samples_in = 0
    self.metrics_out = 0
    self.latencies = []
    self.metric_type = "objective_mobility"
    if self.sample_type != "pressure_bandage":
      self.metric_type += "_" + self.sample_type
    self.app = str(app)

    self.aggregation_buffer = []
    self.aggregation_limit = 100
    self.aggregation_period = 0.05
    self.analyze_period = 0.05
    self.last_time_analyzed = time.time()

    # This is used to calculate the xy gradient when we have enough x and y angle
    # PIMAP metrics based on the gradient_period.
    self.gradient_period = 5
    self.saved_angle_pmetrics_by_id = defaultdict(list)

    # This is used to calculate movements per minute when we have enough
    # xy gradient PIMAP samples based on the movements_per_min_period.
    self.movements_per_min_period = 3600
    self.saved_gradient_pmetrics_by_id = defaultdict(list)

    # Centroid 0
    # locations (x, y): (0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (3, 2)
    # The corresponding x and y indices for centroid 0.
    self.c0_xlocs = np.array([0, 1, 2, 3, 2, 3])
    self.c0_ylocs = np.array([0, 0, 0, 0, 1, 2])
    # The mean of the x coordinates of all pressure bandage data for centroid 0.
    # We divide by 3.0 to normalize the x and y locations.
    self.c0_mean_x = np.mean(self.c0_xlocs/3.0)
    # The mean of the y coordinates of all pressure bandage data for centroid 0.
    self.c0_mean_y = np.mean(self.c0_ylocs/3.0)

    # Centroid 1
    # locations (x, y): (0, 1), (0, 2), (1, 2), (0, 3), (1, 3)
    self.c1_xlocs = np.array([0, 0, 1, 0, 1])
    self.c1_ylocs = np.array([1, 2, 2, 3, 3])
    self.c1_mean_x = np.mean(self.c1_xlocs/3.0)
    self.c1_mean_y = np.mean(self.c1_ylocs/3.0)

    # Centroid 2
    # locations (x, y): (3, 1), (2, 2), (3, 2), (2, 3), (3, 3)
    self.c2_xlocs = np.array([3, 2, 3, 2, 3, 2])
    self.c2_ylocs = np.array([1, 2, 2, 3, 3, 1])
    self.c2_mean_x = np.mean(self.c2_xlocs/3.0)
    self.c2_mean_y = np.mean(self.c2_ylocs/3.0)

  def analyze(self, pimap_samples):
    """The core interaction of PIMAP Analyze Objective Mobility

    Arguments:
      pimap_samples: A list of pressure_bandage PIMAP samples. PIMAP data that is not
        of type pressure_bandage will be filtered out.

    Returns:
      A list of PIMAP metrics that include x_angle, y_angle, xy_gradient, and
      movements_per_min. Please see the original paper for a detailed description
      of these metrics and how the theory behind them.

    Exceptions:
      TypeError:
        If pimap_samples is not a list.
      ValueError:
        If pimap_samples contains non-PIMAP data.
    """
    if not isinstance(pimap_samples, list):
      raise TypeError("The argument pimap_samples must be a list.")

    valid_pimap_samples = list(map(pu.validate_datum, pimap_samples))
    # If pimap_samples is an empty list we still want to continue, this way we can 
    # still return system_samples.
    if not any(valid_pimap_samples) and len(pimap_samples) != 0:
      raise ValueError("Invalid data in pimap_samples.")

    filtered_pimap_samples = list(filter(lambda x: pu.get_type(x) == self.sample_type,
                                         pimap_samples))
    self.aggregation_buffer.extend(filtered_pimap_samples)
    pimap_metrics = []
    if ((len(self.aggregation_buffer) > self.aggregation_limit or
         time.time() - self.last_time_analyzed > self.aggregation_period) and
        len(self.aggregation_buffer) > 0):
      self.last_time_analyzed = time.time()
      start_time_to_analyze = time.time()

      raw_pressure_bandage_string = list(map(pu.get_sample, self.aggregation_buffer))
      raw_pressure_bandage_dict = np.array(list(map(ast.literal_eval,
                                                    raw_pressure_bandage_string)))
      raw_pressure_bandage = np.array(list(map(lambda x: x["pressure_bandage"],
                                               raw_pressure_bandage_dict)))
      normalized_pressure_bandage = raw_pressure_bandage/self.max_pressure

      # The centroid data format is [x, y, mean_pressure]
      c0 = np.empty((len(normalized_pressure_bandage), 3))
      c0[:, 0] = self.c0_mean_x
      c0[:, 1] = self.c0_mean_y
      # The mean of the pressure values of all pressure bandage data for centroid 0.
      c0[:, 2] = np.mean(normalized_pressure_bandage[:, self.c0_ylocs, self.c0_xlocs],
                         axis=1)
      c1 = np.empty((len(normalized_pressure_bandage), 3))
      c1[:, 0] = self.c1_mean_x
      c1[:, 1] = self.c1_mean_y
      c1[:, 2] = np.mean(normalized_pressure_bandage[:, self.c1_ylocs, self.c1_xlocs],
                         axis=1)
      c2 = np.empty((len(normalized_pressure_bandage), 3))
      c2[:, 0] = self.c2_mean_x
      c2[:, 1] = self.c2_mean_y
      c2[:, 2] = np.mean(normalized_pressure_bandage[:, self.c2_ylocs, self.c2_xlocs],
                         axis=1)

      # We now find the plane that contains the three centroids.
      # Equation of a plane: ax + by + cz = d
      # Normal to a plane: n = [a; b; c]
      vector0 = c1 - c0
      vector1 = c2 - c0
      n = np.cross(vector0, vector1)
      a = n[:, 0]
      b = n[:, 1]
      c = n[:, 2]
      # We could use any of the centroids, but we use centroid 0.
      d = c0[:, 0]*a + c0[:, 1]*b + c0[:, 2]*c

      # We use the equation of the plane to calculate the angle of the x slope and
      # y slope.
      x_angle = 360*np.arctan(-a/c)/(2*np.pi)
      y_angle = 360*np.arctan(-b/c)/(2*np.pi)

      angle_metrics = list(map(lambda x, y: {"x_angle":x, "x_angle_units":"degrees",
                                             "y_angle":y, "y_angle_units":"degrees"},
                               x_angle, y_angle))

      angle_pimap_metrics = list(map(lambda x, y: pu.create_pimap_metric(
                                                               self.metric_type, x, y),
                                     self.aggregation_buffer, angle_metrics))

      # The gradient needs to be filtered by patient and device otherwise the gradient
      # will be calculated with respect to all patients and all devices in the given 
      # pimap_samples.

      for pmetric in angle_pimap_metrics:
        pid = pu.get_patient_id(pmetric)
        did = pu.get_device_id(pmetric)
        self.saved_angle_pmetrics_by_id[(pid, did)].append(pmetric)

      gradient_pimap_metrics = []
      for pid, did in self.saved_angle_pmetrics_by_id:
        saved_angle_pmetrics = self.saved_angle_pmetrics_by_id[(pid, did)]
        if len(saved_angle_pmetrics) == 0:
            continue
        timestamps = list(map(lambda x: float(pu.get_timestamp(x)),
                              saved_angle_pmetrics))
        # I assume timestamps are already sorted.
        if len(saved_angle_pmetrics) >= self.gradient_period:
          angle_metrics = list(map(lambda x: ast.literal_eval(pu.get_metric(x)),
                                   saved_angle_pmetrics))
          x_angle = list(map(lambda x: float(x["x_angle"]), angle_metrics))
          y_angle = list(map(lambda x: float(x["y_angle"]), angle_metrics))
          x_gradient = np.abs(np.gradient(x_angle))
          y_gradient = np.abs(np.gradient(y_angle))
          xy_gradient = np.maximum(x_gradient, y_gradient)
          gradient_metrics = list(map(lambda x: {"xy_gradient":x}, xy_gradient))
          new_pimap_metrics = (list(map(lambda x, y: pu.create_pimap_metric(
                                                           self.metric_type, x, y),
                                        saved_angle_pmetrics, gradient_metrics)))
          gradient_pimap_metrics.extend(new_pimap_metrics)
          # Save gradient pimap metrics for calculating movements per minute.
          self.saved_gradient_pmetrics_by_id[(pid, did)].extend(new_pimap_metrics)
          # Remove saved angle pimap metrics as we have already used them.
          self.saved_angle_pmetrics_by_id[(pid, did)] = []

      # Calculate the movements_per_min when we have saved more xy_gradient
      # metrics by patient and device than the movements_per_min_period.
      movements_per_min_pimap_metrics = []
      #for pid, did in self.saved_gradient_pmetrics_by_id:
      #  saved_gradients = self.saved_gradient_pmetrics_by_id[(pid, did)]
      #  while len(saved_gradients) >= self.movements_per_min_period:
      #    saved_gradients = saved_gradients[:self.movements_per_min_period]
      #    timestamps = list(map(lambda x: float(pu.get_timestamp(x)),
      #                        saved_gradients))
      #    elapsed_time = timestamps[-1] - timestamps[0]
      #    gradient_metrics = list(map(lambda x: ast.literal_eval(pu.get_metric(x)),
      #                                saved_gradients))
      #    xy_gradient = list(map(lambda x: x["xy_gradient"], gradient_metrics))
      #    movements_per_min = 60.0*np.sum(np.array(xy_gradient) > 2)/elapsed_time
      #    movements_per_min_metric = {}
      #    movements_per_min_metric["movements_per_min"] = movements_per_min
      #    timestamp = np.mean(timestamps)
      #    # Create a temporary PIMAP sample that will be used to create the
      #    # movements_per_min PIMAP metric. The relevant features are the patient_id,
      #    # device_id, and timestamp.
      #    temp_pimap_sample = pu.create_pimap_sample("temp", pid, did, "temp", timestamp)
      #    new_pimap_metric = pu.create_pimap_metric(self.metric_type, temp_pimap_sample,
      #                                              movements_per_min_metric)
      #    movements_per_min_pimap_metrics.append(new_pimap_metric)
      #    # Remove 1 saved gradient PIMAP metrics as we are using a sliding window.
      #    del self.saved_gradient_pmetrics_by_id[(pid, did)][0]
      #    saved_gradients = self.saved_gradient_pmetrics_by_id[(pid, did)]

      # Calculate the time to analyze and adjust the aggregation limit as necessary.
      time_to_analyze = time.time() - start_time_to_analyze
      if time_to_analyze > self.analyze_period:
        self.aggregation_limit /= 2
      elif len(self.aggregation_buffer) >= self.aggregation_limit:
        self.aggregation_limit += len(self.aggregation_buffer)

      pimap_metrics.extend(angle_pimap_metrics)
      pimap_metrics.extend(gradient_pimap_metrics)
      pimap_metrics.extend(movements_per_min_pimap_metrics)
      self.aggregation_buffer = []

    timestamps = list(map(lambda x: float(pu.get_timestamp(x)), filtered_pimap_samples))
    self.latencies.extend(time.time() - np.array(timestamps))
    pimap_system_samples = []
    if self.system_samples:
      self.samples_in += len(filtered_pimap_samples)
      self.metrics_out += len(pimap_metrics)
      if time.time() - self.system_samples_updated > self.system_samples_update_period:
        sample_type = "system_samples"
        if self.app != "":
          sample_type += "_" + self.app
        patient_id = "analyze"
        device_id = self.metric_type
        sample = {"throughput_in":(self.samples_in/
                                    (time.time() - self.system_samples_updated)),
                  "throughput_out":(self.metrics_out/
                                     (time.time() - self.system_samples_updated)),
                  "aggregation_limit":self.aggregation_limit,
                  "aggregation":len(self.aggregation_buffer)}
        if len(self.latencies) > 0:
          sample["latency"] = np.mean(self.latencies)
        pimap_system_samples.append(pu.create_pimap_sample(sample_type, patient_id,
                                                           device_id, sample))
        self.system_samples_updated = time.time()
        self.samples_in = 0
        self.metrics_out = 0
        self.latencies = []

    return pimap_metrics + pimap_system_samples
