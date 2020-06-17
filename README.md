This paper focuses on the feature extraction module, assessing various types 
of functions and proposing different approaches to normalization, and similarly
analyzing the effectiveness of each sensor included in inertial measuring units.
The study is therefore designed to demonstrate the importance of human activity 
in real-life situations such as elderly health and care. The main research was 
conducted using a publicly available data kit, called the REALDISP Activity 
Recognition data set, which contains recordings in three scenarios of 17 
individuals performing 33 different activities. This data was collected 
using inertial units connected and anchored on the various parts of a human 
body, with nine inertial units giving measurement into three-dimensional 
units throughout the process. Two key areas of the HAR analytics process 
are present: one is an extractor that gathers the most important data from 
inertial signals. The other is an algorithm that determines the physical 
activity categories.
The task of supervised data analysis is composed of two main parts. One is 
called model selection, which is to select the right combination of learning 
methods and adjusting the hyper-parameters of the model for better results. 
The second part is to estimate the performance of the final model‘s output. 
In general, one of the methods that are used to assess classifiers and monitor 
their hyper-parameters is known as K-fold Cross-Validation. However, in K-fold 
Cross-Validation the data points are assumed to be distributed independently 
and identically to allow random and consistent selection of samples used in 
the training and testing sets. In the Human Activity Recognition datasets, 
however, the samples produced by the same subjects are likely to be correlated 
because of various factors, and thus k-fold cross-validations can overestimate 
the activity recognition performance. This paper focuses on the subject wise 
Cross-Validation method which is the same kind of cross-validation but on 
different subjects rather on automatically split parts.

