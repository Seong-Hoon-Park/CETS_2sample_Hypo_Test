class Scoring():
  """
    Scoring for CETS and Pearson correlation. It collect 
      TruePositive  : Correctly predict correlation
      FalsePositive : Predict E~S but actually E and S is not correlated
      FalseNegative : Fail to predict the existance of correlation
    and give result
      Precision : TruePositive / (TruePositive + FalsePositive)
      Recall    : TruePositive / (TruePositive + FalseNegative)
      F1-score  : 2*TruePositive / (2*TruePositive + FalsePositive + FalseNegative)
  """

  def __init__(self, interpret_label, correlat_type, output, 
                correlat_alg='cets', cal_type='exist'):
    self.interpret_label = interpret_label
    self.correlat_type = correlat_type
    self.output = output
    self.tp, self.fp, self.fn = self.calculate_result(cal_type)
    if correlat_alg == 'pearson':
      self.calculate_result_pearson(cal_type)

  def calculate_result(self, cal_type):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    # calculate result with existence of correlation
    if cal_type == 'exist':
      for i in range(len(self.output)):
        label_i = self.interpret_label[i]
        corre_i = self.correlat_type[i]

        for j in range(len(self.output[i])):
          for k in range(len(self.output[i][j])):
            R = self.output[i][j][k][0]
            T = self.output[i][j][k][2]

            label_cnt = label_i[k].count(j+1)
            if label_cnt <= 0:
              if R:
                false_positive += 1

            else:
              if R:
                true_positive += 1
              else:
                false_negative += 1
    
    # calculate result including direction
    elif cal_type == 'dir':
      for i in range(len(self.output)):
        label_i = self.interpret_label[i]
        corre_i = self.correlat_type[i]

        for j in range(len(self.output[i])):
          for k in range(len(self.output[i][j])):
            R = self.output[i][j][k][0]
            T = self.output[i][j][k][2]

            label_cnt = label_i[k].count(j+1)
            if label_cnt <= 0:
              if R:
                false_positive += 1

            else:
              label_idx = label_i[k].index(j+1)
              if R:
                # if GT is 6, consider only existence
                if corre_i[k][label_idx] == 6:
                  true_positive += 1
                else:
                  # when direction is correct
                  if ((corre_i[k][label_idx] < 3) and (T < 3)) or ((corre_i[k][label_idx] >= 3) and (T >= 3)):
                    true_positive += 1
              else:
                false_negative += 1

    # calculate result including direction and effect type
    else:
      for i in range(len(self.output)):
        label_i = self.interpret_label[i]
        corre_i = self.correlat_type[i]

        for j in range(len(self.output[i])):
          for k in range(len(self.output[i][j])):
            R = self.output[i][j][k][0]
            T = self.output[i][j][k][2]

            label_cnt = label_i[k].count(j+1)
            if label_cnt <= 0:
              if R:
                false_positive += 1

            else:
              label_idx = label_i[k].index(j+1)
              if R:
                # if GT is 6, consider only existence
                if corre_i[k][label_idx] == 6:
                  true_positive += 1
                else:
                  if corre_i[k][label_idx] == T:
                    true_positive +=1
              else:
                false_negative += 1

    return true_positive, false_positive, false_negative


  def calculate_result_pearson(self, cal_type):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    # calculate result with existence of correlation
    if cal_type == 'exist':
      for i in range(len(self.output)):
        label_i = self.interpret_label[i]
        corre_i = self.correlat_type[i]

        for j in range(len(self.output[i])):
          for k in range(len(self.output[i][j])):
            R = self.output[i][j][k][0]
            T = self.output[i][j][k][2]

            label_cnt = label_i[k].count(j+1)
            if label_cnt <= 0:
              if R:
                false_positive += 1

            else:
              if R:
                true_positive += 1
              else:
                false_negative += 1

    elif cal_type == 'effect':
      for i in range(len(self.output)):
        label_i = self.interpret_label[i]
        corre_i = self.correlat_type[i]

        for j in range(len(self.output[i])):
          for k in range(len(self.output[i][j])):
            R = self.output[i][j][k][0]
            T = self.output[i][j][k][2]

            label_cnt = label_i[k].count(j+1)
            if label_cnt <= 0:
              if R:
                false_positive += 1

            else:
              label_idx = label_i[k].index(j+1)
              if R:
                # if GT is 6, consider only existence
                if corre_i[k][label_idx] == 6:
                  true_positive += 1
                else:
                  if ((corre_i[k][label_idx] == 1 or corre_i[k][label_idx] == 4) and (T == 1)) \
                          or ((corre_i[k][label_idx] == 2 or corre_i[k][label_idx] == 5) and (T == 2)):
                    true_positive += 1
              else:
                false_negative += 1


  # calculate precision
  def precision(self):
    return self.tp / (self.tp + self.fp)

  # calculate recall
  def recall(self):
    return self.tp / (self.tp + self.fn)

  # calculate f1-score
  def f1_score(self):
    return 2*self.tp / (2*self.tp + self.fp + self.fn)


