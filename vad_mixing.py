import scipy.io.wavfile
import scipy.signal
import os
import glob
import math
import random
import numpy as np
from joblib import Parallel, delayed
import audiotools
import time
import shutil


impRespPath = '/mnt/tatooine/data/impulse_responses/16kHz/wavs16b'
noisePath = '/mnt/tatooine/data/noise/noise_equal_concat/train'
#pocketMovementWav = '/mnt/naboo/noise/noise_equal_concat/trainPocketMovement.wav'
#silenceWav = '/mnt/naboo/noise/silence.wav'
rootPathCleanCorpus = '/mnt/tatooine/data/vad_speaker_recog/TIMIT_Buckeye/vad'
replay_gain_ref_path = '/mnt/tatooine/data/ref_pink_16k.wav'

outputPath1 = '/mnt/alderaan/mlteam3/data/assignment2_mixingdata'


#pobability of pocket movement
probOfPocketMov = 0.2
#pobabiliy of noise being added up to the clean file
probOfNoise = 0.8

#max size of break in sec
maxLenSilence = 0.2;
#min size of break in sec
minLenSilence = 0;

numJobs = 20

wantedFs = 16000
#maximum of the SNR in db (snrMaxDecimal = 10)
snrMax = 20
#minimum of the SNR in db (snrMinDecimal = 0.79433)
snrMin = -2

#Define indices for picking impulse responses
irPhoneIndex = 1
irTestPosIndex = 1

###
# End of config
###

noises = []
silence = []
pocketNoise = []
impResps = []

def cacheNoiseFiles():
	wavFileList = glob.glob(os.path.join(noisePath, '*.wav'))
	for wavFile in wavFileList:
		(fs, samples) = scipy.io.wavfile.read(wavFile)
		noises.append(samples)
		print 'Noise file %s read.' % (wavFile)
	
	#(_,pocketNoise) = scipy.io.wavfile.read(pocketMovementWav)
	#(_,silence) = scipy.io.wavfile.read(silenceWav)
	print 'Noise cached in memory.'

'''
def cacheImpulseResponses():
	phones = ['lg', 's3', 's1', 'htc']
	positions = ['oben', 'unten', 'tisch', 'tasche', 'hose1', 'hose2', 'table', 'jacke']
	
	for phone in phones:
		irSignals = []
		for position in positions:
			fileName = 'ir_' + phone + '_' + position + '.wav'
			(_, samples) = scipy.io.wavfile.read(os.path.join(impRespPath, fileName))
			irSignals.append(samples)
		impResps.append(irSignals)
	print 'IRs cached in memory.'
'''
def cacheImpulseResponses():
	for root, dirs, files in os.walk(impRespPath):
		path = root.split('/')
		for file in files:
			if(file.lower().endswith('.wav')):
				(_, samples) = scipy.io.wavfile.read(os.path.join(impRespPath, file))
				impResps.append(samples)
	print 'IRs cached in memory.'



def getRandomFadedNoise(nSamples):
	while(True):
		index = random.randint(0, len(noises)-1)
		if len(noises[index] >= nSamples):
			#this noise file is long enough, use it
			break
	
	noiseSignal = noises[index]
	
	#get random start point
	rangePotStartPoints = len(noiseSignal) - nSamples
	startInd = math.ceil(random.random() * rangePotStartPoints)
	noiseSegment = noiseSignal[startInd:startInd+nSamples]
	
	#fade to avoid artifacts
	fadeLength = min(len(noiseSegment), 2000)/2
	noiseSegment[:fadeLength] *= np.linspace(0, 1, num=fadeLength)
	noiseSegment[-fadeLength:] *= np.linspace(1, 0, num=fadeLength)
	
	return noiseSegment

'''
def getRandomFadedPocketNoise(nSamples):
	
	#get random start point
	rangePotStartPoints = len(pocketNoise) - nSamples
	if (rangePotStartPoints < 0):
		return np.zeros(nSamples)
	
	startInd = math.ceil(random.random() * rangePotStartPoints)
	noiseSegment = pocketNoise[startInd:startInd+nSamples]

	#fade to avoid artifacts
	fadeLength = min(len(noiseSegment), 2000)/2
	noiseSegment[:fadeLength] *= np.linspace(0, 1, num=fadeLength)
	noiseSegment[-fadeLength:] *= np.linspace(1, 0, num=fadeLength)

	return noiseSegment


def getFadedSilence(nSamples):

	#get random start point
	rangePotStartPoints = len(silence) - nSamples
	if (rangePotStartPoints < 0):
		return np.zeros(nSamples)

	startInd = math.ceil(random.random() * rangePotStartPoints)
	silenceSegment = silence[startInd:startInd+nSamples]

	#fade to avoid artifacts
	fadeLength = min(len(silenceSegment), 2000)/2
	silenceSegment[:fadeLength] *= np.linspace(0, 1, num=fadeLength)
	silenceSegment[-fadeLength:] *= np.linspace(1, 0, num=fadeLength)

	return silenceSegment
'''

def mixFilesInSpeakerPath(spInd, folder):  
	speakerPath = os.path.join(rootPathCleanCorpus, folder)
	wavFileList = glob.glob(os.path.join(speakerPath, '*.wav'))
	
	print 'Starting speaker %s...' % (folder)
	
	for (ind,wavFile) in enumerate(wavFileList):
		(fs, samples) = scipy.io.wavfile.read(wavFile)
		samples = samples.astype(np.float64)/65536.0
		#print 'Speech snippet %s read.' % (wavFile)
		
		#read annotation
		with open(wavFile.replace("wav", "ano")) as f:
			anoList = [int(line.rstrip()) for line in list(f)]
		
		if len(anoList) != len(samples):
			print 'Mismatch in size between annotation and track!'
		
		#get replaygain stats of current file
		file_rplgain = list(audiotools.calculate_replay_gain([ \
			audiotools.open(wavFile) \
			]))[0][1]
		
		#calculate gain to ref file and normalize accordingly
		gain = file_rplgain - ref_rplgain
		normSignal = samples * (10**(gain/20.0))
		
		# SILENCE CODE STARTS HERE
		
		#add silence at start and end
		orglen = len(normSignal)
		'''
		randSilenceLenInSec = round(minLenSilence + \
			(maxLenSilence - minLenSilence) * random.random(), 1)
		randSilenceLenInSamples = int(randSilenceLenInSec * fs)

		normSignal = np.concatenate([ \
			getFadedSilence(randSilenceLenInSamples), \
			normSignal, \
			getFadedSilence(randSilenceLenInSamples) \
			])
		
		anoSilence = list(np.zeros((len(normSignal),), dtype=np.int))
		anoSilence[randSilenceLenInSamples:randSilenceLenInSamples+len(anoList)] = anoList
		anoList = anoSilence
		'''
		# End of Silence Code

		if (random.random() < probOfNoise):
			#mix with noise of same size
			noise = getRandomFadedNoise(len(normSignal))
			#calculate the random SNR
			randomSNR = snrMin + (snrMax-snrMin) * random.random()
			#amplify signal by reducing noise
			noise /= 10**(randomSNR/20) #normSignal *= 10**(randomSNR/20);
			normSignal += noise
		
		# POCKET NOISE CODE BEGINS HERE
		'''
		if (random.random() < probOfPocketMov):
			#mix with noise of same size
			noise = getRandomFadedPocketNoise(len(normSignal))
			#calculate the random SNR
			randomSNR = snrMin + (snrMax-snrMin) * random.random()
			noise /= 10**(randomSNR/20)
			normSignal += noise
		'''
		# End of Pocket Noise Code	
		'''
		randPhone = random.randint(0, len(impResps)-1)
		#exclude target phone from training
		while (randPhone == irPhoneIndex):
			randPhone = random.randint(0, len(impResps)-1)
		
		randPos = random.randint(0, len(impResps[irPhoneIndex])-1)
		#exclude target position from training
		while (randPos == irTestPosIndex):
			randPos = random.randint(0, len(impResps[irPhoneIndex])-1)
		
		irTrain = impResps[randPhone][randPos]
		
		print type(normSignal)
		print "ex : "+ str(len(impResps[irPhoneIndex]))
		print "normSignal Length: "+str(len(normSignal))
		print "irTrain Length: "+str(irTrain.shape)
		
		convolvedSignal1 = scipy.signal.fftconvolve(normSignal, irTrain)[:len(normSignal)]
		'''
		convolvedSignal1 = normSignal

		if not os.path.exists(outputPath1):
		    os.makedirs(outputPath1)
		
		outputFile1 = os.path.join(outputPath1, 'cleanNoiseMix_' \
			+ str(spInd) + '_' + str(ind) + '.wav')
		#shutil.copyfile(wavFile.replace("wav", "ano"), outputFile1.replace("wav", "ano"))
		#print 'Writing %s.' % (outputFile)
		scipy.io.wavfile.write(outputFile1, wantedFs, convolvedSignal1)
		
		f = open(outputFile1.replace("wav", "ano"),'w')
		for (ind,line) in enumerate(anoList):
			if ind == (len(anoList) - 1):
				#no \n at end of file
				f.write("%i" % (line))
			else:
				f.write("%i\n" % (line))
		f.close()
	
	print 'Speaker %s done' % (folder)



if __name__ == '__main__':
	cacheNoiseFiles()
	cacheImpulseResponses()
	
	#replaygain val of reference file
	ref_rplgain = list(audiotools.calculate_replay_gain([ \
		audiotools.open(replay_gain_ref_path) \
		]))[0][1]
	
	#get folder names (folders = speakers)
	all_speaker_names = os.walk(rootPathCleanCorpus).next()[1]
	print '%d speakers detected.' % (len(all_speaker_names))
	
	#for (ind,speaker) in enumerate(all_speaker_names):
	#	mixFilesInSpeakerPath(ind,speaker) 
	results = Parallel(n_jobs=numJobs)(delayed(mixFilesInSpeakerPath)(ind,speaker) \
		for (ind,speaker) in enumerate(all_speaker_names))
	print 'All done.'