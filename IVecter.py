import facereclib.utils as utils
import bob
import numpy
import shutil

# parameters
training_threshold = 5e-4
variance_threshold = 5e-4
max_iterations = 500
relevance_factor = 4             # Relevance factor as described in Reynolds paper
gmm_enroll_iterations = 1        # Number of iterations for the enrollment phase
subspace_dimension_of_t = 400    # This is the dimension of the T matrix. It is tipically equal to 400. 
INIT_SEED = 5489

def load_gmm_stats_list(input_ubm_file, train_features):
  # load UBM
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(input_ubm_file))
  # Projecting training data
  gmm_stats = []
  for feature in train_features:
    # Initializes GMMStats object
    stats = bob.machine.GMMStats(ubm.shape)
    # Accumulate
    ubm.acc_statistcs(feature, stats)
    gmm_stats.append(stats)
  return gmm_stats

def train_ivector(train_features, input_ubm_file):
  # load UBM
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(input_ubm_file))

  # load GMM stats from training files
  gmm_stats = load_gmm_stats_list(input_ubm_file, train_features)  

  # Training IVector enroller
  output_file = 'model/enroller_ivector.hdf5'

  print "IVector training"
  # Perform IVector initialization
  ivector_machine = bob.machine.IVectorMachine(ubm, subspace_dimension_of_t) 
  ivector_machine.variance_threshold = variance_threshold

  # Creates the IVectorTrainer and trains the ivector machine
  ivector_trainer = bob.trainer.IVectorTrainer(update_sigma=True, convergence_threshold=variance_threshold, max_iterations=max_iterationss)
  ivector_trainer.train(ivector_machine, gmm_stats)
  utils.ensure_dir(os.path.dirname(output_file))
  ivector_machine.save(bob.io.HDF5File(output_file, 'w'))
  print "IVector training: saved enroller's IVector machine base to '%s'" % output_file


