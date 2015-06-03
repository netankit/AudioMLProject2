import scipy.io.wavfile as wav
import numpy as np
import math
from features import mfcc
import sys
import os
import cPickle
import scipy.sparse
import glob

if len(sys.argv)!=5:
    print '\nUsage: FrameLevelAnnotation_speakerubm.py <annotations_dir> <noise_mix_speech_dir> <mfcc_vector_output_directory> <class_label_vector_directory>'
    sys.exit()

annotation_dir = sys.argv[1]
audio_files_dir= sys.argv[2]
mfcc_vector_output_dir=sys.argv[3]
class_label_vector_output_dir=sys.argv[4]

if not os.path.exists(mfcc_vector_output_dir):
	os.makedirs(mfcc_vector_output_dir)
if not os.path.exists(class_label_vector_output_dir):
	os.makedirs(class_label_vector_output_dir)


def getframelevelanno(noise_mix_speech_ano_file):
	with open(noise_mix_speech_ano_file, 'rb') as infile1:
		InputData = cPickle.load(infile1)
	InputDataTemp = np.array(scipy.sparse.coo_matrix((InputData),dtype=np.float64).toarray()).tolist()
	infile1.close()
	return InputDataTemp


def getMfccVector(noise_mix_speech_file):
	with open(noise_mix_speech_file, 'rb') as infile1:
		InputData = cPickle.load(infile1)
	InputDataTemp = np.array(scipy.sparse.coo_matrix((InputData),dtype=np.float64).toarray())
	infile1.close()
	return InputDataTemp

def getDataset(annotation_dir, audio_files_dir):
	#print "Invoking getDataset ..... "
	#print str(audio_files_dir)
	wavFileList = glob.glob(os.path.join(audio_files_dir, '*.dat'))
	speech_vector_final = np.zeros((1,13))
	speech_vector_final = np.delete(speech_vector_final, (0), axis=0)
	class_label_vector_final = []
	#print wavFileList
	for (ind,file) in enumerate(wavFileList):
		annotationfilenametmp1= os.path.splitext(os.path.basename(file))[0].replace("mfcc_", "class_label_")
		#print "TMP: "+str(annotationfilenametmp1)
		annotationfilename = str(annotationfilenametmp1)+'.dat'
		#print "ANO-DIR: "+str(annotation_dir)
		#print "ANO-FNAME: "+str(annotationfilename)

		annotation_file_fullpath = os.path.join(annotation_dir,annotationfilename)
		audio_file_fullpath = file

		#print "Annotation File: "+ annotation_file_fullpath
		#print "Audio file:"+ audio_file_fullpath

		mfcc_vector =  getMfccVector(audio_file_fullpath)
		class_label_vector = getframelevelanno(annotation_file_fullpath)

		#print "*Speech Vector:"+str(mfcc_vector.shape)
		#print "*Class Labels:"+str(len(class_label_vector))

		speech_vector_final = np.vstack((speech_vector_final,mfcc_vector))
		class_label_vector_final.extend(class_label_vector)
	return (speech_vector_final,class_label_vector_final)




#Main Routine
print "Start of the Program ....."
all_speaker_names = os.walk(audio_files_dir).next()[1]

for (ind,speaker) in enumerate(all_speaker_names):
	print "Current Speaker: " +str(speaker)
	annotation_dir_speaker = os.path.join(annotation_dir,speaker)
	audio_files_dir_speaker = os.path.join(audio_files_dir,speaker)

	(speech_vector_final,class_label_vector_final) = getDataset(annotation_dir_speaker, audio_files_dir_speaker)

	# Generate the various output files
	tmpmfccfilename = str('mfcc_')+str(speaker)+str('.dat')
	tmpclassfilename = str('class_label_')+str(speaker)+str('.dat')

	mfcc_vector_output_file = os.path.join(mfcc_vector_output_dir, tmpmfccfilename)
	print mfcc_vector_output_file
	class_label_vector_output_file = os.path.join(class_label_vector_output_dir,tmpclassfilename)
	print class_label_vector_output_file

	#MFCC Speech
	mfcc_vector_file = open(mfcc_vector_output_file, 'w')
	temp1 = scipy.sparse.coo_matrix(speech_vector_final)
	cPickle.dump(temp1,mfcc_vector_file,-1)
	mfcc_vector_file.close()

	#Class Labels
	class_label_vector_file = open(class_label_vector_output_file, 'w')
	class_label_vector_final_array = np.array(class_label_vector_final).reshape(len(class_label_vector_final),1)
	temp2 = scipy.sparse.coo_matrix(class_label_vector_final_array)
	cPickle.dump(temp2,class_label_vector_file,-1)
	class_label_vector_file.close()

	print "Final Shapes:"
	print "Speech Vector:"+str(speech_vector_final.shape)
	print "Class Labels:"+str(class_label_vector_final_array.shape)



print "Program completed Successfully!!!"


