import os
import bob
import numpy

ivector_file = '/mnt/alderaan/mlteam3/Assignment2/data/ivectors_25/ubm_ivectors'
whitening_enroler_file = 'model/whitening.hdf5'
wccn_enroller_file = 'model/wccn.hdf5'
lda_enroller_file = 'model/lda.hdf5'
pca_enroller_file = 'model/pca.hdf5'
plda_enroller_file = 'model/plda.hdf5'

SUBSPACE_DIMENSION_OF_F = 5
SUBSPACE_DIMENSION_OF_G = 2


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

def train_whitening_enroller(train_files):
	""" i-vector preprocessing: training whitening enroller"""

	ivectors_matrix = read_ivectors(train_files)
	# create a Linear Machine     # Runs whitening (first method)
	whitening_machine = bob.machine.LinearMachine(ivectors_matrix.shape[1],ivectors_matrix.shape[1])

	# create the whitening trainer
	t = bob.trainer.WhiteningTrainer()

	t.train(whitening_machine, ivectors_matrix)

	# Save the whitening linear machine
	print("Saving the whitening machine..")
	whitening_machine.save(bob.io.HDF5File(whitening_enroler_file, "w"))

	return whitening_machine

def project_whitening(whitening_machine, sample):
	""" i-vector preprocessing: projecting whitening """
	whitened_sample =  whitening_machine.forward(sample)
	return whitened_sample

def train_lda_enroller(train_files):
	""" i-vector preprocessing: training lda enroller"""

	ivectors_matrix = read_ivectors(train_files)

	# create the FisherLDATrainer

	# If set to True, use the pseudo-inverse to calculate
	# and then perform eigen value decomposition (using LAPACK's dgeev)
	# instead of using (the more numerically stable) LAPACK's dsyvgd to 
	# solve the generalized symmetric-definite eigenproblem of the form S_b v=(\lambda) S_w v
	t = bob.trainer.FisherLDATrainer(use_pinv=True) 
	# RuntimeError: The LAPACK function 'dsygvd' returned a non-zero value. 
	# This might be caused by a non-positive definite B matrix.

	LDA_machine, __eig_vals = t.train(ivectors_matrix)

	# Save the whitening linear machine
	print("Saving the LDA machine..")
	LDA_machine.save(bob.io.HDF5File(lda_enroller_file, "w"))

	return LDA_machine

def project_lda(LDA_machine, sample):
	""" i-vector preprocessing: projecting lda """
	projected_sample =  LDA_machine.forward(sample)
	return projected_sample

def train_wccn_enroller(train_files):
	""" i-vector preprocessing: training Within-Class Covariance Normalisation enroller"""
	ivectors_matrix = read_ivectors(train_files)
	#print type(ivectors_matrix[1])
	#print ivectors_matrix[1].shape
	# create the whitening trainer
	t = bob.trainer.WCCNTrainer()
	# Trains the LinearMachine to perform the WCCN, given a training set.
	wccn_machine = t.train(ivectors_matrix)
	
	# Save the whitening linear machine
	print("Saving the wccn machine..")
	wccn_machine.save(bob.io.HDF5File(wccn_enroller_file, "w"))

	return wccn_machine

def project_wccn(whitening_machine, sample):
	""" i-vector preprocessing: projecting wccn"""

	withened_sample =  whitening_machine.forward(sample)
	return withened_sample

def train_pca(training_features):
	"""Trains and returns a LinearMachine that is trained using PCA"""
	data_list = []
	for client in training_features:
		for feature in client:
			data_list.append(feature)
	data = numpy.vstack(data_list)
	t = bob.trainer.PCATrainer()
	machine, __eig_vals = t.train(data)
	# limit number of pcs
	# machine.resize(machine.shape[0], subspace_dimension_pca)
	return machine

def perform_pca_client(machine, client):
	"""Perform PCA on an array"""
	client_data_list = []
	for feature in client:
		# project data
		projected_feature = numpy.ndarray(machine.shape[1], numpy.float64)
		machine(feature, projected_feature)
		# add data in new array
		client_data_list.append(projected_feature)
	client_data = numpy.vstack(client_data_list)

	return client_data

def perform_pca(machine, training_set):
	"""Perform PCA on data"""
	data = []
	for client in training_set:
		client_data = perform_pca_client(machine, client)
		data.append(client_data)
	return data

def train_plda_enroller(train_files):

	# load GMM stats from training files
	training_features = read_ivectors(train_files)

	# train PCA and perform PCA on training data
	pca_machine = train_pca(training_features)
	training_features = perform_pca(pca_machine, training_features)

	input_dimension = training_features[0].shape[1]

	print("Training PLDA base machine")
	# create trainer
	t = bob.trainer.PLDATrainer()
	# train machine
	plda_base = bob.machine.PLDABase(input_dimension, SUBSPACE_DIMENSION_OF_F, SUBSPACE_DIMENSION_OF_G)
	t.train(plda_base, training_features)

	# write machines to file
	proj_hdf5file = bob.io.HDF5File(plda_enroller_file, "w")
	proj_hdf5file.create_group('/pca')
	proj_hdf5file.cd('/pca')
	pca_machine.save(proj_hdf5file)
	proj_hdf5file.create_group('/plda')
	proj_hdf5file.cd('/plda')
	plda_base.save(proj_hdf5file)
	print "saved plda machines"


#ivectors_matrix = read_ivectors(ivector_file)
#whitening = train_whitening_enroller(ivector_file)
lda = train_lda_enroller(ivector_file)
#data = map(lda.forward, ivectors_matrix)

#train_wccn_enroller(ivector_file)



#print("Training PLDA base machine")
#input_dimension = ivectors_matrix[0].shape[1]
# create trainer
#t = bob.trainer.PLDATrainer()
# train machine
#plda_base = bob.machine.PLDABase(input_dimension, SUBSPACE_DIMENSION_OF_F, SUBSPACE_DIMENSION_OF_G)
#t.train(plda_base, ivectors_matrix)

#train_plda_enroller(ivector_file)


