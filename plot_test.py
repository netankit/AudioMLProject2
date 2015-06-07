import matplotlib.pyplot as plt

#SpeakerModel
# Linear Scores
num_gaussian = [64,128,256]
detection_percentage = [40,40,40]
lines = plt.plot(num_gaussian, detection_percentage,marker='o', label='linear' )
plt.setp(lines, color='r', linewidth=1.0)
#plt.axis([0, 300, 0, 100])

# Log Likelihood
num_gaussianLL = [64,128,256]
detection_percentageLL = [70,70,60]
linesLL = plt.plot(num_gaussianLL, detection_percentageLL,marker='x',label='Log-Likelihood' )
plt.setp(linesLL, color='g', linewidth=1.0)

plt.axis([0, 300, 0, 100])
plt.xlabel('Number of Gaussians')
plt.ylabel('Speaker Detection Percentage')
plt.title('Speaker Identification (Linear vs Log Likelihood Scoring)')
plt.legend()
plt.show()