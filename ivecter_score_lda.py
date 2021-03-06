import os
import bob
import numpy
import math
import bob.blitz
from itertools import *
import pandas

whitening_enroler_file = 'model/whitening.hdf5'
wccn_enroller_file = 'model/wccn.hdf5'
lda_enroller_file = 'model/lda.hdf5'
pca_enroller_file = 'model/pca.hdf5'
plda_enroller_file = 'model/plda.hdf5'

# INPUT DIRECTORY
ubm_ivectors_path  = '/mnt/alderaan/mlteam3/Assignment2/data/ivectors_25/ubm_ivectors'
train_ivectors_path = '/mnt/alderaan/mlteam3/Assignment2/data/ivectors_25/speaker_train_ivectors'
test_ivectors_path = '/mnt/alderaan/mlteam3/Assignment2/data/ivectors_25/speaker_test_ivectors'

speaker_list = ["MTQC0.ivec","s32.ivec","FSJG0_f.ivec","MCCS0.ivec","s20_f.ivec","MPMB0.ivec","MAHH0.ivec","s37_f.ivec","s01_f.ivec","s06.ivec"]
#speaker_list = ["s32.ivec","s20_f.ivec","s37_f.ivec","s01_f.ivec","s06.ivec"]

def read_ivectors(ivector_file):
	ivectors_matrix = []
	for root, dir, files in os.walk(ivector_file):
	    for file in files:
	        ivec_path = os.path.join(root, str(file))
	        ivec = bob.io.HDF5File(ivec_path)
	        ivector = ivec.read('ivec')
	        ivector = numpy.array(ivector)
	        #ivectors_matrix = numpy.append(ivectors_matrix, ivector)
	        ivectors_matrix.append(ivector)
	#ivectors_matrix = numpy.vstack(ivectors_matrix)

	#ivectors_matrix = ivectors_matrix.reshape(len(ivectors_matrix)/400, 400)
	return ivectors_matrix

def train_lda_enroller(train_files):
	""" i-vector preprocessing: training lda enroller"""

	ivectors_matrix = read_ivectors(train_files)
	t = bob.trainer.FisherLDATrainer(use_pinv=True) 
	LDA_machine, __eig_vals = t.train(ivectors_matrix)
	# Save the whitening linear machine
	print("Saving the LDA machine..")
	LDA_machine.save(bob.io.HDF5File(lda_enroller_file, "w"))
	return LDA_machine

def project_lda(LDA_machine, sample):
	""" i-vector preprocessing: projecting lda """
	projected_sample =  LDA_machine.forward(sample)
	return projected_sample

def cosine_distance(a, b):
	if len(a) != len(b):
		raise ValueError, "a and b must be same length"
	numerator = sum(tup[0] * tup[1] for tup in izip(a,b))
	denoma = sum(avalue ** 2 for avalue in a)
	denomb = sum(bvalue ** 2 for bvalue in b)
	result = numerator / (numpy.sqrt(denoma)*numpy.sqrt(denomb))
	return result

def cosine_score(client_ivectors, probe_ivector):
	"""Computes the score for the given model and the given probe using the scoring function"""
	scores = []
	for ivec in client_ivectors:
		scores.append(cosine_distance(ivec, probe_ivector))
	return numpy.max(scores)


# lda
lda_machine = train_lda_enroller(ubm_ivectors_path)

results = []
for probe in speaker_list:
	probe_ivectors_path = os.path.join(test_ivectors_path, probe)
	print 'Probe Speaker', probe
	probe_ivec = bob.io.HDF5File(probe_ivectors_path)
	probe_ivectors = probe_ivec.read('ivec')
	probe_ivectors = numpy.array(probe_ivectors)
	for probe_ivector in probe_ivectors:
		for model in speaker_list:
			print probe.split('.',1)[0], ' vs. ', model.split('.',1)[0]
			model_ivec_path = os.path.join(train_ivectors_path, model)
			model_ivec = bob.io.HDF5File(model_ivec_path)
			model_ivectors = model_ivec.read('ivec')
			model_ivectors_projected = project_lda(lda_machine, model_ivectors)
			probe_ivector_projected = project_lda(lda_machine, probe_ivector)
			score = cosine_score(model_ivectors, probe_ivector)
			results += [(probe.split('.',1)[0], model.split('.',1)[0], score)]
			print "SCORE: ", score

print results
df = pandas.DataFrame(results)
df.columns = ["test", "train", "score"]
df.to_csv("results/ivector_lda.csv")
