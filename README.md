# Implementation and Evaluation of Statistical Hypothesis Testing for Network Performance Analysis
Incident diagnosis in network services has emerged as a crucial topic to minimize service delays or failures and provide high-level service quality. In a previous report, we compared and analyzed CETS (Correlating Events with Time Series for Incident Diagnosis), which aims to analyze the correlation between continuous time series data and temporal events, and studies on Root Cause Analysis (RCA) such as FluxRank and ϵ-Diagnosis. In this study, we aim to implement CETS, which has demonstrated satisfactory accuracy and can identify the direction and monotonic effects of correlations, and to verify its performance.

[Documents](https://github.com/Seong-Hoon-Park/CETS_2sample_Hypo_Test/blob/018970cd40434f7de98df8b6daea6d663b11e4e8/Correlating_Events_with_Time_Series_Implementation_and_Evaluation.pdf)

Special thanks to Sangkyu Park in Network analytics Group, Samsung Electronics.

# Architecture
![image](https://github.com/Seong-Hoon-Park/CETS_2sample_Hypo_Test/assets/73863511/7804ad35-679f-40bb-beda-9816fc39f2ec)

* cets.py
  * Main implementation of CETS. Detect sub-series length, two-sample hypothesis test with NN, test effect type.
* data_loader.py
  * Data loading, processing, plotting.
* pearson.py
  * Implementation of PC algorithm as comparison.
* score.py
  * Scoring for CETS and PC.

More details of architecture and trouble-shooting can be found in Documents and Portfolio.
## Reference
[1] Tie-Yan Liu et al. Learning to rank for information retrieval. Foundations and Trends® in Information Retrieval, 3(3):225–331, 2009.

[2] Ping Liu, Yu Chen, Xiaohui Nie, Jing Zhu, Shenglin Zhang, Kaixin Sui, Ming Zhang, and Dan
Pei. Fluxrank: A widely-deployable framework to automatically localizing root cause machines
for software service failure mitigation. In 2019 IEEE 30th International Symposium on Software
Reliability Engineering (ISSRE), pages 35–46. IEEE, 2019.

[3] Huasong Shan, Yuan Chen, Haifeng Liu, Yunpeng Zhang, Xiao Xiao, Xiaofeng He, Min Li, and
Wei Ding. ϵ-diagnosis: Unsupervised and real-time diagnosis of small-window long-tail latency in
large-scale microservice platforms. In The World Wide Web Conference, pages 3215–3222, 2019.

[4] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection
for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th
ACM SIGKDD international conference on knowledge discovery & data mining, pages 2828–2837,
2019.

[5] TsingHuasuya. Omnianomaly. https://github.com/NetManAIOps/OmniAnomaly, 2021.

[6] David A Dickey and Wayne A Fuller. Distribution of the estimators for autoregressive time series
with a unit root. Journal of the American statistical association, 74(366a):427–431, 1979.

[7] Toni Giorgino. Computing and visualizing dynamic time warping alignments in r: the dtw package.
Journal of statistical Software, 31:1–24, 2009.

[8] David MW Powers. Evaluation: from precision, recall and f-measure to roc, informedness, markedness and correlation. arXiv preprint arXiv:2010.16061, 2020.
