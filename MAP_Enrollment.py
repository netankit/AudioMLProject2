import facereclib.utils as utils
import bob
# import bob.io
import numpy
import shutil
import cPickle
import scipy.sparse

# parameters for the GMM
training_threshold = 5e-4
variance_threshold = 5e-4
# parameters of the GMM enrollment
relevance_factor = 4         # Relevance factor as described in Reynolds paper
gmm_enroll_iterations = 1    # Number of iterations for the enrollment phase
INIT_SEED = 5489

model_features_input = '/mnt/alderaan/mlteam3/Assignment2/data/speaker_vaded/speaker0_mfcc.dat'
probe_features_input = '/mnt/alderaan/mlteam3/Assignment2/data/speaker_vaded/speaker0_mfcc.dat'
output_model_file = 'model/model_speaker0_new.hdf5'
output_probe_file = 'model/probe_speaker0_new.hdf5'


# read UBM
ubm_file = 'model/gmm_256g.hdf5'
ubm_hdf5 = bob.io.HDF5File(ubm_file)
ubm = bob.machine.GMMMachine(ubm_hdf5)
ubm.set_variance_thresholds(variance_threshold)

# read model features - MFCC Features for the Speaker(s)
with open(model_features_input, 'rb') as test_file:
	model_features = cPickle.load(test_file)
model_features = scipy.sparse.coo_matrix((model_features),dtype=numpy.float64).toarray()
test_file.close()

# read probe features - MFCC Features for the Speaker(s)
# with open(probe_features_input, 'rb') as test_file:
# 	probe_features = cPickle.load(test_file)
# probe_features = scipy.sparse.coo_matrix((probe_features),dtype=numpy.float64).toarray()
# test_file.close()
probe_features = model_features

# prepare MAP_GMM_Trainer
MAP_GMM_trainer = bob.trainer.MAP_GMMTrainer(relevance_factor = relevance_factor, update_means = True, update_variances = False, update_weights=False)
rng = bob.core.random.mt19937(INIT_SEED)
MAP_GMM_trainer.set_prior_gmm(ubm)

# Enrolls a GMM using MAP adaptation of the UBM, given a list of 2D numpy.ndarray's of feature vectors"""
# We can perform this for all the 10 speakers
gmm = bob.machine.GMMMachine(ubm)
gmm.set_variance_thresholds(variance_threshold)
MAP_GMM_trainer.train(gmm, model_features)  #, gmm_enroll_iterations, training_threshold, rng
gmm.save(bob.io.HDF5File(output_model_file, 'w'))

# Probe is to get statistics seperately from UBM and model gmm, and then compare, So this will contain the list of all speaker files "sp" directory.
probes = []  #waiting implement

# Computes GMM statistics for the given probe feature vector against a UBM, given an input 2D numpy.ndarray of feature vectors
# Initializes GMMStats
gmm_stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)
# Accumulates statistics
ubm.acc_statistics(probe_features, gmm_stats)
probe = gmm_stats
probe.save(bob.io.HDF5File(output_probe_file, 'w'))
print 'Done'

# Computes the score for the given model and the given probe using the scoring function from the config file
#print bob.machine.linear_scoring([model], ubm, [probe], [], frame_length_normalisation = True)
# this scoring works, but don't know if it is correct!
# Maybe, use the probe log_likelihood of ubm and enrolled gmm to compare the score!
