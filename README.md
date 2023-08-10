# Anomaly-Detection-in-a-Recommender-System

This project is about detecting anomalies in a recommender system based on ratings data from a streaming video dataset. The task is to identify instances where users may be sharing accounts or engaging in other behavior that deviates from typical usage patterns.

Examples of anomalous behavior we may encounter include:

* **Account sharing:** This occurs when multiple users access a single account to watch videos, which can lead to skewed ratings and viewing histories.
* **Promoting items:** Some users may artificially promote or demote certain videos by rating them higher or lower than they would normally, either to manipulate the system or for personal reasons.
* **Review bombing:** A group of users may band together to artificially lower the rating of a particular video or video series, leading to a distorted view of its popularity.
* **Rating consistency:** Some users may rate all videos in a similar manner, regardless of their actual preferences or viewing habits. This may be an indication of fake accounts or bots.
* **Abnormal viewing patterns:** Users may exhibit unusual viewing patterns, such as watching an abnormally high number of videos in a short period of time, or repeatedly watching the same video multiple times. This may indicate account sharing or fraudulent activity.
  

# Steps of the project

The project consists of two steps. 

* In the first step, we will provide labels for all instances in the dataset, indicating which are anomalies and which are not. This step is important because it allows participants to train their models on a fully labeled dataset and establish a baseline for performance.

* In the second step, we will provide a partially labeled dataset, meaning that some instances will be labeled as anomalies, but others will be left unlabeled. This simulates a semi-supervised setting, where participants are required to identify anomalies without full knowledge of the labels.

It is important to note that the dataset may contain different types of anomalies, meaning that participants may observe anomalies in the second dataset that were not observed in the first. This requires participants to be flexible in their approach and able to adapt their models to new types of anomalies.

To submit their results, participants will be provided with two separate sets of data, one for each step of the competition. The leaderboard will consider the F1 score of participants' models on a subset of the instances in the dataset. The final evaluation, however, will consider all instances in the dataset. This ensures that participants are not simply overfitting to a specific subset of the data, but rather are building models that can generalize to new instances.
