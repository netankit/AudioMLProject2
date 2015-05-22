import numpy as np
import time
import sys
import cPickle
import scipy.sparse
import os

speaker_list = ["speaker0_mfcc.dat", "speaker1_mfcc.dat", "speaker2_mfcc.dat", "speaker3_mfcc.dat", "speaker4_mfcc.dat", "speaker5_mfcc.dat", "speaker6_mfcc.dat", "speaker7_mfcc.dat", "speaker8_mfcc.dat", "speaker9_mfcc.dat"]

classlabel_list = ["speaker0_class_label.dat", "speaker1_class_label.dat", "speaker2_class_label.dat", "speaker3_class_label.dat", "speaker4_class_label.dat",
                   "speaker5_class_label.dat", "speaker6_class_label.dat", "speaker7_class_label.dat", "speaker8_class_label.dat", "speaker9_class_label.dat"]

original_speaker_path = '/mnt/alderaan/mlteam3/Assignment2/data/sp/'
output_vaded_path = '/mnt/alderaan/mlteam3/Assignment2/data/speaker_vaded/'
class_label_path = original_speaker_path

count = int(0)

for file in speaker_list:
    print 'Current File: ' + str(file)
    with open(os.path.join(original_speaker_path, file), 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()

    # CLASS LABEL FILE
    classlabel_filename = os.path.join(class_label_path, classlabel_list[count])
    with open(classlabel_filename, 'rb') as infile2:
        label = cPickle.load(infile2)
    infile2.close()

    label = np.array(scipy.sparse.coo_matrix((label), dtype=np.int16).toarray()).tolist()
    label = map(str, label)
    label = [int(i.strip('[').strip(']')) for i in label]
    mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()

    ones_index = [i for i, j in enumerate(label) if j == 1]
    ones = [mfcc[i] for i in ones_index]

    print mfcc.shape
    print len(ones)

    mfcc_vaded = open(os.path.join(output_vaded_path, file), 'w')
    temp1 = scipy.sparse.coo_matrix(ones)
    cPickle.dump(temp1, mfcc_vaded, -1)
    mfcc_vaded.close()
    count = count + int(1)
