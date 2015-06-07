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

#speaker_list = ["MTQC0.ivec","s32.ivec","FSJG0_f.ivec","MCCS0.ivec","s20_f.ivec","MPMB0.ivec","MAHH0.ivec","s37_f.ivec","s01_f.ivec","s06.ivec"]
speaker_list = ["s32.ivec","s20_f.ivec","s37_f.ivec","s01_f.ivec","s06.ivec"]


def load_plda_enroller(plda_enroller_file):
	"""Reads the PCA projection matrix and the PLDA model from file"""
	# read UBM
	proj_hdf5file = bob.io.HDF5File(plda_enroller_file)
	proj_hdf5file.cd('/pca')
	pca_machine = bob.machine.LinearMachine(proj_hdf5file)
	proj_hdf5file.cd('/plda')
	plda_base = bob.machine.PLDABase(proj_hdf5file)
	plda_machine = bob.machine.PLDAMachine(plda_base)
	return pca_machine, plda_base

def perform_pca_client(pca_machine, client):
	"""Perform PCA on an array"""
	client_data_list = []
	for feature in client:
		# project data
		projected_feature = numpy.ndarray(pca_machine.shape[1], numpy.float64)
		projected_feature = pca_machine(feature)
		# add data in new array
		client_data_list.append(projected_feature)
	client_data = numpy.vstack(client_data_list)
	return client_data
	
def plda_enroll(pca_machine, plda_base, enroll_features):
	"""Enrolls the model by computing an average of the given input vectors"""
	#enroll_features = numpy.vstack(enroll_features)
	#enroll_features_projected = perform_pca_client(pca_machine, enroll_features)
	features =[ numpy.array([feature]) for feature in enroll_features]
	plda_trainer = bob.trainer.PLDATrainer()
	plda_trainer.train(plda_base, features)
	return plda_base

def read_plda_model(model_file):
	"""Reads the model, which in this case is a PLDA-Machine"""
	# read machine and attach base machine
	print ("model: %s" %model_file)
	plda_machine = bob.machine.PLDAMachine(bob.io.HDF5File(str(model_file)), plda_base)
	return plda_machine

def plda_score(model, probe):
	return model.compute_log_likelihood(probe)


pca_machine, plda_base = load_plda_enroller(plda_enroller_file)

# plda
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
			plda_base = plda_enroll(pca_machine, plda_base, model_ivectors)
			plda_machine = bob.machine.PLDAMachine(plda_base)
			score = plda_score(plda_machine, probe_ivector)
			results += [(probe.split('.',1)[0], model.split('.',1)[0], score)]
			print "SCORE: ", score

print results
df = pandas.DataFrame(results)
df.columns = ["test", "train", "score"]
df.to_csv("results/ivector_plda.csv")
