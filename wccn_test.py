import os
import bob
import numpy

ivector_file = '/mnt/alderaan/mlteam3/Assignment2/model/ubm_ivectors'
whitening_enroler_file = 'model/whitening.hdf5'
wccn_enroller_file = 'model/wccn.hdf5'
lda_enroller_file = 'model/lda.hdf5'
pca_enroller_file = 'model/pca.hdf5'
plda_enroller_file = 'model/plda.hdfa'


def read_ivectors(ivector_file):
	ivectors_matrix  = []
	for root, dir, files in os.walk(ivector_file):
	    for file in files:
	        ivec_path = os.path.join(root, str(file))
	        ivec = bob.io.HDF5File(ivec_path)
	        ivector = ivec.read('ivec')
	        ivector = numpy.array(ivector)
	        ivectors_matrix.append(ivector)
	# ivectors_matrix = numpy.vstack(ivectors_matrix)
	return ivectors_matrix

def train_wccn_enroller(data):
	""" i-vector preprocessing: training Within-Class Covariance Normalisation enroller"""
	#ivectors_matrix = read_ivectors(train_files)
	#print type(ivectors_matrix[1])
	#print ivectors_matrix[1].shape
	# create the whitening trainer
	t = bob.trainer.WCCNTrainer()
	# Trains the LinearMachine to perform the WCCN, given a training set.
	wccn_machine = t.train(data)
	
	# Save the whitening linear machine
	print("Saving the wccn machine..")
	wccn_machine.save(bob.io.HDF5File(wccn_enroller_file, "w"))

	return wccn_machine

def project_wccn(whitening_machine, sample):
	""" i-vector preprocessing: projecting wccn"""

	withened_sample =  whitening_machine.forward(sample)
	return withened_sample


# ok 
data = [numpy.array([[ 1.2622, -1.6443, 0.1889], [ 0.4286, -0.8922, 1.3020]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377], [-0.8718, -0.4788, 0.3988]]), 
		numpy.array([[-0.0098, -0.3121,-0.1807],  [ 0.4301,  0.4886, -0.1456]])]

# RuntimeError: The LAPACK dgetrf function returned a non-zero value.
data1 = [numpy.array([[-0.0098, -0.3121,-0.1807],[-0.0098, -0.3121,-0.1807]]),
		numpy.array([[-0.0098, -0.3121,-0.1807],[-0.0098, -0.3121,-0.1807]]),
		numpy.array([[-0.0098, -0.3121,-0.1807],[-0.0098, -0.3121,-0.1807]])]

# RuntimeError: The LAPACK dgetrf function returned a non-zero value.
data2 = [numpy.array([[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377],[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377],[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889]]), 
		numpy.array([[-0.0098, -0.3121,-0.1807],[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889]])]

# OK
data21 = [numpy.array([[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377],[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889]]), 
		numpy.array([[-0.1613,  0.0430, 0.6377],[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889]]), 
		numpy.array([[-0.0098, -0.3121,-0.1807],[ 1.2622, -1.6443, 0.1889],[ 1.2622, -1.6443, 0.1889]])]


# RuntimeError: The LAPACK dgetrf function returned a non-zero value.
data3 = [numpy.array([[ 1.2622, -1.6443, 0.1889, 1.2622], [ 1.2622, -1.6443, 0.1889,  0.0430]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377,  0.0430], [ 1.2622, -1.6443, 0.1889,  0.0430]]), 
		numpy.array([[-0.0098, -0.3121,-0.1807,  0.0430], [ 1.2622, -1.6443, 0.1889,  0.0430]])]

# RuntimeError: The LAPACK dgetrf function returned a non-zero value.
data4 = [numpy.array([[ 1.2622, -1.6443, 1.2622, -1.6443], [ 0.4286, -0.8922, 1.2622, -1.6443]]), 
		numpy.array([[-0.6613,  0.0430, 1.2622, -1.6443], [-0.8718, -0.4788,1.2622, -1.6443 ]]), 
		numpy.array([[-0.6613,  0.0430, 1.2622, -1.6443], [-0.8718, -0.4788,1.2622, -1.6443 ]]), 
		numpy.array([[-0.6613,  0.0430, 1.2622, -1.6443], [-0.8718, -0.4788,1.2622, -1.6443 ]]), 
		numpy.array([[-0.0098, -0.3121,1.2622, -1.6443],  [ 0.4301,  0.4886, 1.2622, -1.6443]])]
# ok
data5 = [numpy.array([[ 1.2622, -1.6443], [ 0.4286, -0.8922]]), 
		numpy.array([[-0.6613,  0.0430], [-0.8718, -0.4788]]), 
		numpy.array([[-0.6613,  0.0430], [-0.8718, -0.4788]]), 
		numpy.array([[-0.6613,  0.0430], [-0.8718, -0.4788]]), 
		numpy.array([[-0.0098, -0.3121],  [ 0.4301,  0.4886]])]

data6 = [numpy.array([[ 1.2622, -1.6443, 0.1889, 1.2622], [ 1.2622, -1.6443, 0.1889,  0.0430]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377,  0.0430], [ 1.2622, -1.6443, 0.1889,  0.0430]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377,  0.0430], [ 1.2622, -1.6443, 0.1889,  0.0430]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377,  0.0430], [ 1.2622, -1.6443, 0.1889,  0.0430]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377,  0.0430], [ 1.2622, -1.6443, 0.1889,  0.0430]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377,  0.0430], [ 1.2622, -1.6443, 0.1889,  0.0430]]), 
		numpy.array([[-0.6613,  0.0430, 0.6377,  0.0430], [ 1.2622, -1.6443, 0.1889,  0.0430]]), 
		numpy.array([[-0.0098, -0.3121,-0.1807,  0.0430], [ 1.2622, -1.6443, 0.1889,  0.0430]])]

#wccn = train_wccn_enroller(ivectors_matrix)

print data.shape
wccn = train_wccn_enroller(data)
