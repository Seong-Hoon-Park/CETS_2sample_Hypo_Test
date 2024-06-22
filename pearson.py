import numpy as np

class Pearson():
  def __init__(self, time_series, event_sequences, p = 0.1):
    """
      time_series     : list of multivariate time series
      event_sequences : list of event_sequences
      p               : testing threshold value
    """
    self.time_series = self.normalize_time_series(time_series)
    self.event_sequences = event_sequences
    self.p = p

  # normalize time series
  def normalize_time_series(self, time_series):
    time_series_tmp = []

    for ts in time_series:
      time_series_tmp.append((ts - np.min(ts)) / (np.max(ts) - np.min(ts)))

    return time_series_tmp

  # test pearson correlation for each time_series and event
  def run_pearson(self):
    output_list = []

    for i in range(len(self.time_series)):
      output_each_ts = []

      for j in range(self.time_series[i].shape[1]):
        output_each_dim = []

        for k in range(len(self.event_sequences[i])):
          R, D, T = self.pc_one_series_and_event(self.time_series[i][:,j], 
                                          self.event_sequences[i][k])
          output_each_dim.append([R, D, T])

        output_each_ts.append(output_each_dim)

      output_list.append(output_each_ts)

    return output_list

  # actually test pearson correlation
  def pc_one_series_and_event(self, series, event):
    R = False
    T = 0

    # pre-process event
    ev_ts = np.zeros(len(series))
    for et in event:
      ev_ts[et] = 1

    series_std = np.std(series)
    ev_ts_std = np.std(ev_ts)
    p_es = 0

    # calculate pearson correlation coefficient
    if series_std * ev_ts_std > 0:
      p_es = np.cov(series, ev_ts)[0][1] / np.std(series) / np.std(ev_ts)

    # test correlation
    if p_es > self.p:
      R = True
      T = 1
    elif p_es < -self.p:
      R = True
      T = 2

    return R, 0, T
  