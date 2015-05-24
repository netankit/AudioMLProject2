import facereclib.utils as utils
import bob
# import bob.io
import numpy
#import shutil
import cPickle
import scipy.sparse
import random
import os
print "Start..."
# parameters for the GMM
training_threshold = 5e-4
variance_threshold = 5e-4
# parameters of the GMM enrollment
relevance_factor = 4         # Relevance factor as described in Reynolds paper
gmm_enroll_iterations = 1    # Number of iterations for the enrollment phase
INIT_SEED = 5489

#model_features_input = '/mnt/alderaan/mlteam3/Assignment2/data/speaker_vaded/speaker0_mfcc.dat'
#probe_features_input = '/mnt/alderaan/mlteam3/Assignment2/data/speaker_vaded/speaker0_mfcc.dat'
#output_model_file = 'model/model_speaker0.hdf5'
#output_probe_file = 'model/probe_speaker0.hdf5'

model_features_path = '/mnt/alderaan/mlteam3/Assignment2/data/speaker_vaded'

model_output_path = '/mnt/alderaan/mlteam3/Assignment2/output/map_adapted_model'
model_probe_output_path = '/mnt/alderaan/mlteam3/Assignment2/output/probe_files'

#speaker_vaded_list = ["speaker0_mfcc.dat","speaker1_mfcc.dat","speaker2_mfcc.dat","speaker3_mfcc.dat","speaker4_mfcc.dat","speaker5_mfcc.dat","speaker6_mfcc.dat","speaker7_mfcc.dat","speaker8_mfcc.dat","speaker9_mfcc.dat"]

#speaker_vaded_list = ["speaker0_mfcc.dat","speaker1_mfcc.dat","speaker2_mfcc.dat","speaker3_mfcc.dat","speaker4_mfcc.dat"]

speaker_vaded_list = ['speaker0_mfcc.dat', 'speaker1_mfcc.dat']

# read UBM
ubm_file = 'model/gmm_256g.hdf5'
ubm_hdf5 = bob.io.HDF5File(ubm_file)
ubm = bob.machine.GMMMachine(ubm_hdf5)
ubm.set_variance_thresholds(variance_threshold)


def getModelFeatures(model_features_input):
    # read model features - MFCC Features for the Speaker(s)
    with open(model_features_input, 'rb') as infile:
        model_features = cPickle.load(infile)
    infile.close()
    model_features_arr = scipy.sparse.coo_matrix((model_features), dtype=numpy.float64).toarray()
    return model_features_arr

def fixname(filenamestr):
    tmp = filenamestr[:-3]
    return tmp+str("hdf5")

# prepare MAP_GMM_Trainer
MAP_GMM_trainer = bob.trainer.MAP_GMMTrainer(relevance_factor=relevance_factor, update_means=True, update_variances=False, update_weights=False)
rng = bob.core.random.mt19937(INIT_SEED)
MAP_GMM_trainer.set_prior_gmm(ubm)

# Computes the score for the given model and the given probe using the scoring function from the config file
# TODO: UNCOMMENT HERE
#randomProbeFile = os.path.join(model_probe_output_path, random.choice(speaker_vaded_list))
randomProbeFile = bob.machine.GMMMachine(bob.io.HDF5File(os.path.join(model_probe_output_path, 'speaker1_mfcc.hdf5')))
print "Random Probe Chosen: "+ str(randomProbeFile)
models_list = []

for model in speaker_vaded_list:
    models_list.append[bob.machine.GMMMachine(bob.io.HDF5File(os.path.join(model_output_path, fixname(model))))]

#modelfile = bob.io.HDF5File(os.path.join(model_output_path, fixname(model)))
#print "Speaker: "+str(fixname(model))
score = bob.machine.linear_scoring(models_list, ubm, [randomProbeFile], [], frame_length_normalisation = True)
print str(type(score))
print score
# this scoring works, but don't know if it is correct!
# Maybe, use the probe log_likelihood of ubm and enrolled gmm to compare the score!
