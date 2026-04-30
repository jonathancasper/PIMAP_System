"""PIMAP Analyze component that analyzes Sentinel bandage data.

PIMAP Analyze converts SentinelBandage PIMAP samples into SentinelHeatMapMetrics. This
component depends on the sentineltoolkit.

License:-
Author: Eric Vin
"""
import pickle

from pimap import pimaputilities as pu
from sentineltoolkit.sentinel_analysis import analyze_sample_discrete_heatmap, analyze_sample_continuous_heatmap, analyze_sample_triangle_heatmap

class PimapAnalyzeHeatmap:
  """ Converts a calibrated sentinel bandage scan into a heatmap. """
  def __init__(self, frequency, analysis_type):
    # Which frequency to display on the heatmap
    self.frequency = int(frequency)
    # Which heatmap type: 0 for discrete, 1 for continuous
    self.analysis_type = analysis_type

  def analyze(self, pimap_samples):
    pimap_metrics = []

    for pimap_sample in pimap_samples:
      pickled_scan = pu.get_data(pimap_sample).encode()

      sensor_data = pickle.loads(pickled_scan)

      print("Sample Timestamp: ", pu.get_timestamp(pimap_sample))

      # Discrete analysis shows each connection as its own block on the heatmap
      if self.analysis_type == 0:
          heatmap_data = analyze_sample_discrete_heatmap(sensor_data, self.frequency)
      # Continuous analysis averages neigbouring connections, to produce a smoother
      # heatmap
      elif self.analysis_type == 1:
          heatmap_data = analyze_sample_continuous_heatmap(sensor_data, self.frequency)
      elif self.analysis_type == 2:
          heatmap_data = analyze_sample_triangle_heatmap(sensor_data, self.frequency)
      else:
          assert False

      pimap_metric = pickle.dumps(heatmap_data, 0).decode()

      pimap_metric = pu.create_pimap_metric("SentinelHeatmap", pimap_sample, pimap_metric)

      pimap_metrics.append(pimap_metric)

      return pimap_metrics
