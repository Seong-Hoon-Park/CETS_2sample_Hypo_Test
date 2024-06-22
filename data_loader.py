import numpy as np
import os
import sys
from matplotlib import pyplot as plt

class DataLoader():
  def __init__(self, dataset_folder):
    self.dataset_folder = dataset_folder

  # load time series for 28 machines of 38-dim
  def load_time_series(self, category):
    path = os.path.join(self.dataset_folder, category)
    file_list = os.listdir(path)
    file_list.sort()
    time_series = []

    for filename in file_list:
      if filename.endswith('.txt'):
        tmp = np.genfromtxt(os.path.join(path, filename),
                          dtype=np.float32,
                          delimiter=',')
        time_series.append(tmp)

    return time_series

  # load events in label format
  def load_events_in_label_format(self, category, interpret_label, correlat_type):
    path = os.path.join(self.dataset_folder, category)
    file_list = os.listdir(path)
    file_list.sort()
    event_sequences = []
    interpret_label_u = []  # updated label after event grouping
    correlation_type_u = [] # updated correlation type after event grouping

    j = 0
    for filename in file_list:
      if filename.endswith('.txt'):
        tmp = np.genfromtxt(os.path.join(path, filename),
                        dtype=np.float32)
        event_seq = []
        
        # find change points
        for i in range(0, len(tmp)-1):
          if tmp[i] == 0 and tmp[i+1] == 1:
            event_seq.append(i)

        # group events which have same interpretation label
        event_dict = {}
        label_dict = {}
        corre_dict = {}
        for i in range(len(interpret_label[j])):
          it_label = ' '.join(str(s) for s in interpret_label[j][i])
          label_dict[it_label] = interpret_label[j][i]
          corre_dict[it_label] = correlat_type[j][i]
          if it_label in event_dict:
            event_dict[it_label].append(event_seq[i])
          else:
            event_dict[it_label] = [event_seq[i]]

        _event_seq = []
        _interp_label = []
        _corre_type = []
        for l, e in event_dict.items():
          _event_seq.append(e)
          _interp_label.append(label_dict[l])
          _corre_type.append(corre_dict[l])
        
        event_sequences.append(_event_seq)
        interpret_label_u.append(_interp_label)
        correlation_type_u.append(_corre_type)
        j = j+1
    
    return event_sequences, interpret_label_u, correlation_type_u

  # load interpretation labels
  # FORMAT
  # ANOMALY_START_TIME-ANOMALY_END_TIME:(dim1),(dim2),(dim3)...
  # (dimi) is a index of dimension
  def load_interpret_label(self, category):
    path = os.path.join(self.dataset_folder, category)
    file_list = os.listdir(path)
    file_list.sort()
    interpretation_labels = []

    for filename in file_list:
      if filename.endswith('.txt'):
        f = open(os.path.join(path, filename), 'r')
        labels = []

        while True:
          line = f.readline()
          if not line:
            break
          
          tmp = line[line.find(':')+1:len(line)-1].split(',')
          tmp_int = list(map(int, tmp))
          labels.append(tmp_int)

        interpretation_labels.append(labels)
    
    return interpretation_labels

  # load correlation type
  # FORMAT
  # (type1),(type2),(type3)...
  # (typei) is a index of dimension
  #     0 for E ->  S
  #     1 for E +-> S
  #     2 for E --> S
  #     3 for S ->  E
  #     4 for S +-> E
  #     5 for S --> E
  #     6 for E  ~  S
  def load_correlation_type(self, category):
    path = os.path.join(self.dataset_folder, category)
    file_list = os.listdir(path)
    file_list.sort()
    correlation_types = []

    for filename in file_list:
      if filename.endswith('.txt'):
        f = open(os.path.join(path, filename), 'r')
        types = []

        while True:
          line = f.readline()
          if not line:
            break
          
          tmp = line.split(',')
          tmp_int = list(map(int, tmp))
          types.append(tmp_int)

        correlation_types.append(types)
    
    return correlation_types


class DataPlotter():
  def __init__(self, figsize, x_visible, x_len):
    self.figsize = figsize
    self.x_visible = x_visible
    self.x_len = x_len

  # plot time series
  def plt_time_series(self, time_series):
    plt.figure(figsize=self.figsize)

    for i in range(len(time_series)):
      ts = time_series[i]
      plt.subplot(len(time_series), 1, i+1) 
      plt.xlim([0, self.x_len])
      plt.gca().axes.xaxis.set_visible(self.x_visible)
      plt.ylabel('X' + str(i+1), labelpad=15, fontdict={'size': 18})
      plt.yticks([np.min(ts), np.max(ts)], fontsize=12)

      # plot data
      plt.plot(ts[:self.x_len, :9], 'lightgray')
      plt.plot(ts[:self.x_len, 10:19], 'darkgray')
      plt.plot(ts[:self.x_len, 20:29], 'dimgray')
      plt.plot(ts[:self.x_len, 30:], 'k')

    plt.show()


  # plot time series with events (in red line)
  def plt_time_series_with_events(self, time_series, event_sequences):
    plt.figure(figsize=self.figsize)

    for i in range(len(time_series)):
      ts = time_series[i]
      evs = event_sequences[i]
      
      # convert event to time series
      ev_ts = np.zeros(len(time_series[i]))
      for ev in evs:
        for et in ev:
          ev_ts[et] = 1

      plt.subplot(len(time_series), 1, i+1)
      plt.xlim([0, self.x_len])
      plt.gca().axes.xaxis.set_visible(self.x_visible)
      plt.ylabel('X' + str(i+1), labelpad=15, fontdict={'size': 18})
      plt.yticks([np.min(ts), np.max(ts)], fontsize=12)

      # plot data
      plt.plot(ts[:self.x_len, :9], 'lightgray')
      plt.plot(ts[:self.x_len, 10:19], 'darkgray')
      plt.plot(ts[:self.x_len, 20:29], 'dimgray')
      plt.plot(ts[:self.x_len, 30:], 'k')
      plt.plot(ev_ts[:self.x_len], 'r')

    plt.show()

    


