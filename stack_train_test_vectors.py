import numpy as np
import time
import sys
import cPickle
import scipy.sparse
import os
import math
from features import mfcc

dataset_dir = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/'



# INPUT DIRECTORY

# SPEAKER_MODEL
train_features_path = os.path.join(dataset_dir,'train_vaded','speaker_model')
test_features_path =  os.path.join(dataset_dir,'test_vaded','speaker_model')

# UBM
#train_features_path = os.path.join(dataset_dir,'train_vaded','ubm')
#test_features_path = os.path.join(dataset_dir,'test_vaded','ubm')


# OUTPUT DIRECTORY
# SPEAKER_MODEL
train_output_path = os.path.join(dataset_dir,'train_vaded','speaker_model_stacked')
test_output_path = os.path.join(dataset_dir,'test_vaded','speaker_model_stacked')

# UBM
#train_output_path = os.path.join(dataset_dir,'train_vaded','ubm_stacked')
#test_output_path = os.path.join(dataset_dir,'test_vaded','ubm_stacked')

if not os.path.exists(train_output_path):
    os.makedirs(train_output_path)
if not os.path.exists(test_output_path):
    os.makedirs(test_output_path)

all_speaker_names = os.walk(train_features_path).next()[1]

def getModelFeatures(model_features_input):
    # read model features - MFCC Features for the Speaker(s)
    with open(model_features_input, 'rb') as infile:
        model_features = cPickle.load(infile)
    infile.close()
    #print model_features
    model_features_arr = scipy.sparse.coo_matrix((model_features), dtype=np.float64).toarray()
    #print model_features_arr
    return model_features_arr

def getstackedVector(speaker_path):
    stacked_vector_final = np.zeros((1,13))
    stacked_vector_final = np.delete(stacked_vector_final, (0), axis=0)
    #print "Original : "+str(stacked_vector_final.shape)
    tmp = []
    for files in os.walk(speaker_path):
        tmp = files[2]

    for mfccfile in tmp:
        file_to_read = os.path.join(speaker_path,mfccfile)
        #print file_to_read
        utterance_vector = getModelFeatures(file_to_read)
        #print utterance_vector
        #print "utterance_vector : "+str(utterance_vector.shape)
        (X,Y) = utterance_vector.shape
        if not Y is 13:
            continue;
        else:
            stacked_vector_final = np.vstack((stacked_vector_final,utterance_vector))
    return stacked_vector_final

for speaker in all_speaker_names:
    #if speaker.startswith('s'):
    #TRAIN
    speaker_path = os.path.join(train_features_path,speaker)
    stacked_vector_final = getstackedVector(speaker_path)
    print "Final : "+str(stacked_vector_final.shape)
    mfcc_vector_output_file = os.path.join(train_output_path,str(speaker)+'_train_stacked.dat')
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(stacked_vector_final)
    cPickle.dump(temp1,mfcc_vector_file,-1)
    mfcc_vector_file.close()

    #TEST
    speaker_path = os.path.join(test_features_path,speaker)
    stacked_vector_final = getstackedVector(speaker_path)
    mfcc_vector_output_file  = os.path.join(test_output_path, str(speaker)+'_test_stacked.dat')
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(stacked_vector_final)
    cPickle.dump(temp1,mfcc_vector_file,-1)
    mfcc_vector_file.close()

print "Done!"


