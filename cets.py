import numpy as np
import math
import random
from prompt_toolkit.formatted_text.utils import fragment_list_len
from statsmodels.tsa.stattools import adfuller
from scipy.signal import find_peaks
import dtw as DTW

class CETS():
  """
    CETS(Correlating Events with Time Series in "Correlating Event 
    with Time Series for Incident Diagnosis") is an algorithm for
    discovering correlation between events and time series.
    It regard the correlation discovery as two-sample hypothesis test
    with test statistics from Nearest Neighbor algorithm.
    For T_r,p = (the proportion of pairs containing two sub-series
    from the same sample, among all pairs formed by a sample sub-series
    and one of its neares neighbors in the pooled sample Z),
    (pr)^(1/2)(T_r,p-mean_r)/(std_r) ~ N(0, 1).

  """

  def __init__(self, time_series, event_sequences, 
                sub_length = 0, ts_ratio= 1, acf_ratio = 16, 
                adf_ratio = 64, width_ratio = 0.5, sub_len_max = 100, 
                sub_len_min = 20, r = 3, alpha = 1.96, nn_dis_threshold = 0.0025):
    """
      time_series     : list of multivariate time series.
      event_sequences : list of event_sequences.
      sub_length      : sub-series length k as integer. 0 for auto detecting k.
                        with Dickey-Fuller unit root test and autocorrelation.
      ts_ratio (>= 1)   : use 1/ts_ratio part of time-series for detecting length.
      acf_ratio (>= 1)  : use 1/acf_ratio part of time-series for acf.
      adf_ratio (>= 1)  : use 1/adf_ratio part of time-series for adf.
      width_ratio       : use 1/width_ratio of lag for peak width.
                          lag is guessed with Dickey-Fuller unit root test.
      sub_len_max       : sub-series length maximum.
      sub_len_min       : sub-series length manimum.
      r                 : use r-NN for two-sample test r=ln(p) is proved to be 
                          a good choice when p is big enough.
      alpha             : threshold for testing (1.96 for P = 0.0025).
      nn_dis_threshold  : nearest neighbor distance threshold. if distance
                          between two series is smaller than this value, they
                          are regarded as same value.
    """
    self.time_series = self.normalize_time_series(time_series)
    self.event_sequences = event_sequences
    self.sub_len_max = sub_len_max
    self.sub_len_min = sub_len_min
    self.sub_length_list = self.init_sub_length(sub_length, ts_ratio, acf_ratio, adf_ratio, width_ratio)
    self.r = r
    self.alpha = alpha
    self.nn_dis_threshold = nn_dis_threshold

  # initialize sub-series length
  def init_sub_length(self, sub_length, ts_ratio, acf_ratio, adf_ratio, width_ratio):
    if sub_length > 0:
      return [[sub_length] * len(self.time_series[i].shape[1]) 
                for i in range(len(self.time_series))]
      
    elif sub_length == 0:
      sub_length_list = []
      for ts in self.time_series:

        # auto detect sub-series length for each dimension
        sub_length_dim = []
        for d in range(ts.shape[1]):
          sub_length_dim.append(self.auto_detect_sub_length(ts[:int(len(ts)/ts_ratio),d], 
                                acf_ratio, adf_ratio, width_ratio))

        for i in range(len(sub_length_dim)):
          if sub_length_dim[i] == 0:
            sub_length_dim[i] = int(sum(sub_length_dim) / len(sub_length_dim))

        sub_length_list.append(sub_length_dim)

      return sub_length_list

    else:
      raise ValueError('sub_length must be natural or 0 for auto-detection.')

  # normalize time series
  def normalize_time_series(self, time_series):
    time_series_tmp = []

    for ts in time_series:
      time_series_tmp.append((ts - np.min(ts)) / (np.max(ts) - np.min(ts)))

    return time_series_tmp

  # auto correlation function
  def acf(self, series, k):
    return np.mean(series * np.roll(series, k))


  # automatically detect sub-series length with autocorrelation function
  # set sub-series length to first peak of autocorrelation.
  def auto_detect_sub_length(self, ts, acf_ratio, adf_ratio, width_ratio):
    acf_result = [self.acf(ts,k) for k in range(int(len(ts)/acf_ratio))]

    # set find peak width with argumented Dickey-Fuller unit root test
    w = 1
    try:
      adf_result = adfuller(ts[:int(len(ts)/adf_ratio)], regression='ct', autolag='AIC')
      w = adf_result[2]/width_ratio
    except:
      w = 0
 
    peaks, _ = find_peaks(acf_result, width=w)
    
    if len(peaks) == 0:
      return 0
    else:
      # use first peak
      if peaks[0] > self.sub_len_max:
        return self.sub_len_max
      elif peaks[0] < self.sub_len_min:
        return self.sub_len_min
      else:
        return peaks[0]


  # two-sample hypothesis test with Nearest Neighbor algorithm
  def two_sample_test_with_NN(self, sample0, sample1, r, alpha = 1.96):
    pooled_sample = sample0 + sample1   # pooled sample Z
    n0, n1 = len(sample0), len(sample1)
    p = n0 + n1
    l0, l1 = n0 / p, n1 / p             # lambda0, lambda1
    m_r = l0*l0 + l1*l1
    var_r = l0*l1 + 4*(l0*l0)*(l1*l1)

    T_rp = 0
    for i in range(p):
      ind = self.NN_indicator_with_DTW(i, pooled_sample, n0, p, r)
      T_rp += ind
    T_rp = T_rp / r / p

    p_value = math.sqrt(r*p)*(T_rp - m_r) / var_r
    if p_value > alpha:
      return True
    else:
      return False


  # Ir(x, A1, A2) =   1, if x ∈ Ai && NNr(x, A) ∈ Ai,
  #                   0, otherwise.
  def NN_indicator_with_DTW(self, i, Z, n0, p, r):
    x = Z[i]
    dists = []    # list for distances
    ind = 0       # indicator

    for j in range(p):
      if j != i:
        dists.append((j, DTW.dtw(x, Z[j], keep_internals=True).distance))

    dists.sort(key = lambda element : element[1])
    if (dists[-1][1] - dists[0][1]) / 2 / len(x) < self.nn_dis_threshold:
      ind = 0.5*r
    else:
      # return sum of indicator for 1 to r-th NN
      for rr in range(r):
        if (i < n0 and dists[rr][0] < n0) or (i >= n0 and dists[rr][0] >= n0):
          ind += 1
    
    return ind


  # test effect type
  def effect_type_test(self, sample0, sample1, alpha = 1.96):
    effect_type = 0   # offect type = 0 for none, 1 for positive, 2 for negative
    n = len(sample0)
    m0 = np.mean(sample0)
    m1 = np.mean(sample1)
    var0 = np.var(sample0)
    var1 = np.var(sample1)

    if var0 + var1 <= 0:
      t_score = 0
    else:
      t_score = (m0 - m1) / math.sqrt((var0*var0 + var1*var1) / n)

    if t_score > alpha:
      effect_type = 2
    elif t_score < -alpha:
      effect_type = 1

    return effect_type


  # verbose test result
  def verbose_test_result(self, name0, name1, R, T):
    cor_type = ''
    if T == 0:
      cor_type = '<-( )'
    elif T == 1:
      cor_type = '<-(+)'
    elif T == 2:
      cor_type = '<-(-)'
    elif T == 3:
      cor_type = '( )->'
    elif T == 4:
      cor_type = '(+)->'
    else:
      cor_type = '(-)->'

    if R:
      print("{0} {1} {2}".format(name0, cor_type, name1))

  # run CETS for each time_series and event
  def run_cets(self):
    output_list = []

    for i in range(len(self.time_series)):
      output_each_ts = []

      for j in range(self.time_series[i].shape[1]):
        output_each_dim = []

        for k in range(len(self.event_sequences[i])):
          R, D, T = self.cets_one_series_and_event(self.time_series[i][:,j], 
                                          self.event_sequences[i][k],
                                          self.sub_length_list[i][j])
          self.verbose_test_result('Time-series X{0} dim {1}'.format(i+1, j+1), 
                                    'Effect {0}'.format(k+1), R, T)
          output_each_dim.append([R, D, T])

        output_each_ts.append(output_each_dim)

      output_list.append(output_each_ts)

    return output_list


  # actually run CETS algorithm
  def cets_one_series_and_event(self, series, event, k):
    front_sub_series = []   # front sub-series set
    rear_sub_series = []    # rear sub-series set
    rand_sub_series = []    # randomly sampled sub-series set from S

    R = False
    D_f = False
    D_r = False
    T = -1

    # pre-process event
    if event[0] < k:
      del event[0]
    if event[-1]+k >= len(series):
      del event[-1]

    # initialize front, rear, and random sub-series
    for e in event:
      front_sub_series.append(series[e-k:e])
      rear_sub_series.append(series[e+1:e+k+1])
      rand_sub_series.append(np.random.choice(series, k, False))

    # test front sub-series with random sample
    D_f = self.two_sample_test_with_NN(front_sub_series, rand_sub_series,
                                        self.r, self.alpha)
    D_r = self.two_sample_test_with_NN(rear_sub_series, rand_sub_series, 
                                        self.r, self.alpha)

    if D_r and (not D_f):
      R = True
      T = self.effect_type_test(front_sub_series, rear_sub_series, self.alpha)
    elif D_f:
      R = True
      T = self.effect_type_test(front_sub_series, rear_sub_series, self.alpha) + 3

    # output
    # R = True for correlated False for not correlated
    # D = (D_f, D_r), two sample hypothesis test result of front and rear
    # T = correlation testing result
    #     -1 for not correlated
    #     0 for E ->  S
    #     1 for E +-> S
    #     2 for E --> S
    #     3 for S ->  E
    #     4 for S +-> E
    #     5 for S --> E
    return R, (D_f, D_r), T


