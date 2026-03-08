import numpy as np

score=np.array([54,60,65,70,75,75,80,85,91,95])
# mean
mean = np.mean(score)
print(mean) # 75.0
# median
median = np.median(score)
print(median) #75.0
unique, count = np.unique(score, return_counts=True)
print(count) # [1 1 1 1 2 1 1 1 1]
# mode
mode = unique[np.argmax(count)]
print(mode) # 75
# range
range = np.ptp(score)
print(range) # 41
variance = np.var(score, ddof = 1)
print(variance) # 174.66666666666666
std_dev = np.std(score, ddof = 1)
print(std_dev) # 13.21615173439934
q1 = np.percentile(score, 25)
print(q1) # 66.25
q2 = np.percentile(score, 50)
print(q2) # 75.0
q3 = np.percentile(score, 75)
print(q3) # 83.75
iqr = q3 - q1
print(iqr) # 17.5
skwness = (3 * (mean - median)) / std_dev
print(skwness) # 0.0
