import bob
import numpy
import os
import cPickle
import scipy
import sys

if len(sys.argv)!=3:
    print '\nUsage: python scoring_GMM_UBM_final.py <number_of_gaussians> <speaker_model_or_ubm>'
    sys.exit()

num_gaussian = sys.argv[1]
marker = sys.argv[2]

print "Starting Scoring Experiments for GMM - UBM (MAP Enrollment)..."
##
#
# Log Likelihood and Linear Scoring Implemented.
# Cosine Distance has some issues.
#

# parameters for the GMM
variance_threshold = 5e-4

# Output Directory
model_output_dir = '/mnt/alderaan/mlteam3/Assignment2/output/gmm_map_model_final/'
model_probe_output_dir = '/mnt/alderaan/mlteam3/Assignment2/output/gmm_probe_final/'

score_dump_dir = '/mnt/alderaan/mlteam3/Assignment2/output/score_dumps/'


model_output_path = os.path.join(model_output_dir,marker,num_gaussian)
#print model_output_path
model_probe_output_path = os.path.join(model_probe_output_dir,marker,num_gaussian)

if (marker=="speaker_model"):
    # Input Directory
    model_features_path_train = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/train_vaded/speaker_model_stacked/'
    model_features_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/test_vaded/speaker_model_stacked/'
    original_speaker_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/speaker_model/filewise_mfcc/'

if (marker=="ubm"):
    model_features_path_train = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/train_vaded/ubm_stacked/'
    model_features_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/test_vaded/ubm_stacked/'
    original_speaker_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/ubm/filewise_mfcc/'

all_speaker_names = os.walk(original_speaker_path).next()[1]


# FULL LIST
#speaker_vaded_list = ["speaker0_mfcc.dat","speaker1_mfcc.dat","speaker2_mfcc.dat","speaker3_mfcc.dat","speaker4_mfcc.dat","speaker5_mfcc.dat","speaker6_mfcc.dat","speaker7_mfcc.dat","speaker8_mfcc.dat","speaker9_mfcc.dat"]

# TEST LIST
#speaker_vaded_list = ['speaker0_mfcc.dat', 'speaker1_mfcc.dat']

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
    csv_results = []
    for probe in all_speaker_names:
        randomProbeFileName = str(probe)+'.hdf5'
        #print 'Printing Linear Scores (vs All Speakers) for Probe: '+str(randomProbeFileName)
        randomProbeFile = bob.machine.GMMStats(bob.io.HDF5File(os.path.join(model_probe_output_path, randomProbeFileName)))
        indiv_result = []
        for model in all_speaker_names:
            model_and_score = ()
            #print 'SPEAKER:'+ str(model)
            gmm_model_file = bob.machine.GMMMachine(bob.io.HDF5File(os.path.join(model_output_path, str(model)+'.hdf5')))
            models_list = [gmm_model_file]

            #Linear SCORING
            score = calculateLinearScores(models_list, ubm, randomProbeFile)
            #print "SCORE: "+str(score)
            model_and_score = {str(model):float(score)}
            indiv_result.append(model_and_score)
        csv_results.append({str(probe): indiv_result})
    return csv_results


def LogLikelihoodScoreRun():
    csv_results = []
    for testspeaker in all_speaker_names:
        #print 'Printing Log Likelihood Scores (vs All Speakers) for Probe: '+str(testspeaker)
        test_file_path = os.path.join(model_features_path, str(testspeaker) + '_test_stacked.dat')
        test_feature = getModelFeatures(test_file_path)
        indiv_result = []
        for model in all_speaker_names:
            model_and_score = ()
            #print 'SPEAKER:'+ str(model)
            model_file_path = os.path.join(model_output_path, str(model)+'.hdf5')
            gmm_model_file = bob.machine.GMMMachine(bob.io.HDF5File(model_file_path))
            #score = calculateLogLikelihoodScores(models_list, ubm, randomProbeFile)
            score = numpy.mean([gmm_model_file.log_likelihood(row_sample) - ubm.log_likelihood(row_sample)
                for row_sample in test_feature])
            #print "SCORE: "+str(score)
            model_and_score = {str(model):float(score)}
            indiv_result.append(model_and_score)
        csv_results.append({str(testspeaker): indiv_result})
    return csv_results

def cosine_distance(a, b):
    if len(a) != len(b):
        raise ValueError, "a and b must be same length"
    numerator = sum(tup[0] * tup[1] for tup in izip(a,b))
    denoma = sum(avalue ** 2 for avalue in a)
    denomb = sum(bvalue ** 2 for bvalue in b)
    result = numerator / (sqrt(denoma)*sqrt(denomb))
    return result


def CosineScoreRun():
    '''
    pseduo code
    1. get probe's  gmm_stats1 from ubm
    2. get get probe's  gmm_stats2 from model's gmm
    3. cos_distance(gmm_stats1, gmm_stats2)

    '''
    csv_results = []
    for probe in all_speaker_names:
        randomProbeFileName = str(probe)+'.hdf5'
        #print 'Printing Linear Scores (vs All Speakers) for Probe: '+str(randomProbeFileName)
        randomProbeFile = bob.machine.GMMStats(bob.io.HDF5File(os.path.join(model_probe_output_path, randomProbeFileName)))
        indiv_result = []
        for model in all_speaker_names:
            model_and_score = ()
            #print 'SPEAKER:'+ str(model)
            model_features = getModelFeatures(os.path.join(model_features_path_train, str(probe)+'_train_stacked.dat'))
            gmm_stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)

            # Accumulates statistics
            ubm.acc_statistics(model_features, gmm_stats)
            model_gmm_stats = gmm_stats
            print type(model_gmm_stats)
            print type(randomProbeFile)
            score = cosine_distance(model_gmm_stats,randomProbeFile)
            print "SCORE: "+str(score)
            model_and_score = {str(model):float(score)}
            indiv_result.append(model_and_score)
        csv_results.append({str(probe): indiv_result})
    return csv_results



# Run to calculate Linear Scores (vs All Speakers) for Probe:
print 'Computing Linear Scores'
csv_results = LinearScoreRun()
score_output_file = os.path.join(score_dump_dir,'linear_score_'+str(marker)+'_'+str(num_gaussian)+'.score')
score_dump_file = open(score_output_file, 'w')
cPickle.dump(csv_results,score_dump_file,-1)
score_dump_file.close()
#print csv_results
print "Saved Linear Scores to Disk!"

# Run to calculate Log Likelihood (vs All Speakers) for Probe:
# Using: Probe log_likelihood of UBM and enrolled gmm to compare the score!
print 'Computing Log Likelihood Scores'
csv_results = LogLikelihoodScoreRun()
#print csv_results
score_output_file = os.path.join(score_dump_dir,'loglikelihood_score_'+str(marker)+'_'+str(num_gaussian)+'.score')
score_dump_file = open(score_output_file, 'w')
cPickle.dump(csv_results,score_dump_file,-1)
score_dump_file.close()
print "Saved Loglikelihood Scores to Disk!"

'''
print 'Computing Cosine Scores'
csv_results = CosineScoreRun()
score_output_file = os.path.join(score_dump_dir,'cosine_score_'+str(marker)+'_'+str(num_gaussian)+'.score')
score_dump_file = open(score_output_file, 'w')
cPickle.dump(csv_results,score_dump_file,-1)
score_dump_file.close()
#print csv_results
print "Saved Cosine Scores to Disk!"
'''