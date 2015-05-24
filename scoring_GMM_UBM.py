import bob
import numpy
import os
import cPickle
import scipy


print "Starting Scoring Experiments for GMM - UBM (MAP Enrollment)..."
##
#
# Log Likelihood and Linear Scoring Implemented.
#

# parameters for the GMM
variance_threshold = 5e-4

# Input Directory
model_features_path = '/mnt/alderaan/mlteam3/Assignment2/data/speaker_vaded'

# Output Directory
model_output_path = '/mnt/alderaan/mlteam3/Assignment2/output/map_adapted_model'
model_probe_output_path = '/mnt/alderaan/mlteam3/Assignment2/output/probe_files'

# FULL LIST
speaker_vaded_list = ["speaker0_mfcc.dat","speaker1_mfcc.dat","speaker2_mfcc.dat","speaker3_mfcc.dat","speaker4_mfcc.dat","speaker5_mfcc.dat","speaker6_mfcc.dat","speaker7_mfcc.dat","speaker8_mfcc.dat","speaker9_mfcc.dat"]

# TEST LIST
#speaker_vaded_list = ['speaker0_mfcc.dat', 'speaker1_mfcc.dat']

# read UBM
ubm_file = 'model/gmm_256g.hdf5'
ubm_hdf5 = bob.io.HDF5File(ubm_file)
ubm = bob.machine.GMMMachine(ubm_hdf5)
ubm.set_variance_thresholds(variance_threshold)


def fixname(filenamestr):
    '''Fixes the file name extention for the model and probe file, which is in .dat format and needs to be .hdf5 format '''
    tmp = filenamestr[:-3]
    return tmp+str("hdf5")


def getModelFeatures(model_features_input):
    '''Read model features - MFCC Features for the Speaker(s)'''
    with open(model_features_input, 'rb') as infile:
        model_features = cPickle.load(infile)
    infile.close()
    model_features_arr = scipy.sparse.coo_matrix((model_features), dtype=numpy.float64).toarray()
    return model_features_arr

# Calculates Linear Scores
def calculateLinearScores(models_list, ubm, randomProbeFile):
    return bob.machine.linear_scoring(models_list, ubm, [randomProbeFile], [], frame_length_normalisation = True)

# Computes the score for the given model and the given random probe using the scoring function from the config file

def LinearScoreRun():
    for model in speaker_vaded_list:
        randomProbeFileName = fixname(model)
        print 'Printing Linear Scores (vs All Speakers) for Probe: '+str(randomProbeFileName)
        randomProbeFile = bob.machine.GMMStats(bob.io.HDF5File(os.path.join(model_probe_output_path, randomProbeFileName)))

        for model in speaker_vaded_list:
            print 'SPEAKER:'+ str(fixname(model))

            gmm_model_file = bob.machine.GMMMachine(bob.io.HDF5File(os.path.join(model_output_path, fixname(model))))
            models_list = [gmm_model_file]

            #Linear SCORING
            score = calculateLinearScores(models_list, ubm, randomProbeFile)
            print "SCORE: "+str(score)

def LogLikelihoodScoreRun():
    for testspeaker in speaker_vaded_list:
        print 'Printing Log Likelihood Scores (vs All Speakers) for Probe: '+str(testspeaker)
        test_feature = getModelFeatures(os.path.join(model_features_path, testspeaker))
        for model in speaker_vaded_list:
            print 'SPEAKER:'+ str(fixname(model))
            gmm_model_file = bob.machine.GMMMachine(bob.io.HDF5File(os.path.join(model_output_path, fixname(model))))
            #score = calculateLogLikelihoodScores(models_list, ubm, randomProbeFile)
            score = numpy.mean([gmm_model_file.log_likelihood(row_sample) - ubm.log_likelihood(row_sample)
                for row_sample in test_feature])
            print "SCORE: "+str(score)


# Run to calculate Linear Scores (vs All Speakers) for Probe:
print 'Computing Linear Scores'
LinearScoreRun()

# Run to calculate Log Likelihood (vs All Speakers) for Probe:
# Using: Probe log_likelihood of UBM and enrolled gmm to compare the score!
print 'Computing Log Likelihood Scores'
LogLikelihoodScoreRun()


