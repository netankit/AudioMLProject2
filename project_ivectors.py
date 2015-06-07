import os
import bob
import numpy
import shutil
import cPickle
import scipy.sparse
import facereclib.utils as utils
from sklearn import preprocessing

# parameters
training_threshold = 5e-4
variance_threshold = 5e-4
max_iterations = 25
relevance_factor = 4             # Relevance factor as described in Reynolds paper
gmm_enroll_iterations = 1        # Number of iterations for the enrollment phase
subspace_dimension_of_t = 400    # This is the dimension of the T matrix. It is tipically equal to 400. 
INIT_SEED = 5489

# parameters for the GMM
gaussians = 256

KMeans_HDF5 = 'models/kmeans_256G.hdf5'
GMM_HDF5 = 'models/gmm_256g.hdf5'
InputData = 'data/mfcc_vaded.dat'
input_ubm_features = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/ubm_vaded'
input_speaker_features = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/speaker_model_vaded'
input_speaker_train_features = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/train_vaded/speaker_model'
input_speaker_test_features = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/test_vaded/speaker_model'

def normalize(data):
    return preprocessing.normalize(data,norm='l2')

def read_mfcc_features(input_features):

    with open(input_features, 'rb') as file:
        features = cPickle.load(file)
    features = scipy.sparse.coo_matrix((features),dtype=numpy.float64).toarray()
    if features.shape[1] != 0:
        features = normalize(features)
    return features

def load_training_gmmstats(input_features):

    gmm_stats_list = []
    for root, dir, files in os.walk(input_features):
        for file in files:
            features_path = os.path.join(root, str(file))
            features = read_mfcc_features(features_path)
            stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)
            if features.shape[1] == 13:
                ubm.acc_statistics(features, stats)
                gmm_stats_list.append(stats)

    return gmm_stats_list

def train_enroller(input_features):

    # load GMM stats from UBM training files 
    gmm_stats = load_training_gmmstats(input_features)  

    # Training IVector enroller
    output_file = 'model/tv_enroller_25.hdf5'

    print "training enroller (total variability matrix) ", max_iterations, 'max_iterations'
    # Perform IVector initialization with the UBM
    ivector_machine = bob.machine.IVectorMachine(ubm, subspace_dimension_of_t) 
    ivector_machine.variance_threshold = variance_threshold

    # Creates the IVectorTrainer and trains the ivector machine
    ivector_trainer = bob.trainer.IVectorTrainer(update_sigma=True, convergence_threshold=variance_threshold, max_iterations=max_iterations)
    # An trainer to extract i-vector (i.e. for training the Total Variability matrix)
    ivector_trainer.train(ivector_machine, gmm_stats)
    ivector_machine.save(bob.io.HDF5File(output_file, 'w'))
    print "IVector training: saved enroller's IVector machine base to '%s'" % output_file

    return ivector_machine

def lnorm_ivector(ivector):
    norm = numpy.linalg.norm(ivector)
    if norm != 0:
        return ivector/numpy.linalg.norm(ivector)
    else:
        return ivector

def save_ivectors(data, feature_file):
    hdf5file = bob.io.HDF5File(feature_file, "w")
    hdf5file.set('ivec', data)

def project_ivectors(input_features):
    """Extract the ivectors for all files of the database"""
    print "projecting ivetors"
    tv_enroller = bob.machine.IVectorMachine(ubm, subspace_dimension_of_t)
    tv_enroller.load(bob.io.HDF5File("model/tv_enroller_25.hdf5"))
    #print input_features
    for root, dir, files in os.walk(input_features):
        ivectors = []
        for file in files:
            features_path = os.path.join(root, str(file))
            features = read_mfcc_features(features_path)
            stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)
            if features.shape[1] == 13:
                ubm.acc_statistics(features, stats)
                ivector = tv_enroller.forward(stats)   
                lnorm_ivector(ivector)
                ivectors.append(ivector)
        ivectors_path = input_features + '/' + os.path.split(root)[1] + '.ivec'
        print  ivectors_path
        save_ivectors(ivectors, ivectors_path)
        print "saved ivetors to '%s' " % ivectors_path


#############################################

ubm = bob.machine.GMMMachine(bob.io.HDF5File('/mnt/alderaan/mlteam3/Assignment2/models/gmm_256g.hdf5'))
#train_enroller(input_ubm_features)
project_ivectors(input_ubm_features)
project_ivectors(input_speaker_train_features)
project_ivectors(input_speaker_test_features)


