import facereclib.utils as utils
import bob
# import bob.io
import numpy
#import shutil
import cPickle
import scipy.sparse
import random
import os
import sys

if len(sys.argv)!=3:
    print '\nUsage: python MAP_Enrollment_final.py <number_of_gaussians> <speakermodel_or_ubm>'
    sys.exit()


num_gaussian = sys.argv[1]
marker = sys.argv[2]

print "Starting MAP GMM-UBM Enrollment Training for 'n' Speakers..."
# parameters for the GMM
training_threshold = 5e-4
variance_threshold = 5e-4
# parameters of the GMM enrollment
relevance_factor = 4         # Relevance factor as described in Reynolds paper
gmm_enroll_iterations = 1    # Number of iterations for the enrollment phase
INIT_SEED = 5489

dataset_dir = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/'
output_dir = '/mnt/alderaan/mlteam3/Assignment2/output/'

if (marker=="speakermodel"):
    # INPUT DIRECTORY
    # SPEAKER_MODEL
    model_features_path = os.path.join(dataset_dir,'train_vaded','speaker_model_stacked')
    probe_features_path = os.path.join(dataset_dir,'test_vaded','speaker_model_stacked')
    original_speaker_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/speaker_model/filewise_mfcc/'

    ## OUTPUT DIRECTORY
    model_output_path = os.path.join(output_dir, 'gmm_map_model_final', 'speaker_model',num_gaussian)
    model_probe_output_path = os.path.join(output_dir, 'gmm_probe_final', 'speaker_model',num_gaussian)

if (marker=="ubm"):
    # INPUT DIRECTORY
    # UBM
    model_features_path = os.path.join(dataset_dir,'train_vaded','ubm_stacked')
    probe_features_path = os.path.join(dataset_dir,'test_vaded','ubm_stacked')
    original_speaker_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/ubm/filewise_mfcc/'

    ## OUTPUT DIRECTORY
    model_output_path = os.path.join(output_dir, 'gmm_map_model_final', 'ubm', num_gaussian)
    model_probe_output_path = os.path.join(output_dir, 'gmm_probe_final', 'ubm', num_gaussian)


if not os.path.exists(model_output_path):
    os.makedirs(model_output_path)
if not os.path.exists(model_probe_output_path):
    os.makedirs(model_probe_output_path)

# speaker_vaded_list = ["speaker0_mfcc.dat","speaker1_mfcc.dat","speaker2_mfcc.dat","speaker3_mfcc.dat","speaker4_mfcc.dat","speaker5_mfcc.dat","speaker6_mfcc.dat","speaker7_mfcc.dat","speaker8_mfcc.dat","speaker9_mfcc.dat"]

all_speaker_names = os.walk(original_speaker_path).next()[1]

# read UBM
if (num_gaussian=="64"):
    ubm_file = 'model_gmm_ubm/gmm_ubm_64G.hdf5'
if (num_gaussian=="128"):
    ubm_file = 'model_gmm_ubm/gmm_ubm_128G.hdf5'
if (num_gaussian=="256"):
    ubm_file = 'model_gmm_ubm/gmm_ubm_256G.hdf5'
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

# Enrolls a GMM using MAP adaptation of the UBM, given a list of 2D numpy.ndarray's of feature vectors"""
# We can perform this for all the 10 speakers
for speaker in all_speaker_names:
    speaker_filename = os.path.join(model_features_path, str(speaker)+'_train_stacked.dat')
    print 'Speaker File: '+str(speaker_filename)
    model_features = getModelFeatures(speaker_filename)

    output_model_file = os.path.join(model_output_path, str(speaker)+'.hdf5')
    gmm = bob.machine.GMMMachine(ubm)
    gmm.set_variance_thresholds(variance_threshold)
    MAP_GMM_trainer.train(gmm, model_features)  #, gmm_enroll_iterations, training_threshold, rng
    gmm.save(bob.io.HDF5File(output_model_file, 'w'))

# Probe is to get statistics seperately from UBM and model' enrolled gmm, and then compare,
# *So this will contain the list of all speaker files "speaker_vaded" directory.
# Computes GMM statistics for the given probe feature vector against a UBM, given an input 2D numpy.ndarray of feature vectors
# Initializes GMMStats
for probe in all_speaker_names:
    output_probe_file = os.path.join(model_probe_output_path, str(probe)+'.hdf5')

    probe_features = getModelFeatures(os.path.join(probe_features_path, str(probe)+'_test_stacked.dat'))
    gmm_stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)

    # Accumulates statistics
    ubm.acc_statistics(probe_features, gmm_stats)
    probe = gmm_stats
    probe.save(bob.io.HDF5File(output_probe_file, 'w'))

# Handling scoring  in a seperate python file "scoring.py".

print 'Done'