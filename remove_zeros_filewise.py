import numpy as np
import time
import sys
import cPickle
import scipy.sparse
import os
'''
speaker_list = ["speaker0_mfcc.dat", "speaker1_mfcc.dat", "speaker2_mfcc.dat", "speaker3_mfcc.dat", "speaker4_mfcc.dat", "speaker5_mfcc.dat", "speaker6_mfcc.dat", "speaker7_mfcc.dat", "speaker8_mfcc.dat", "speaker9_mfcc.dat"]

classlabel_list = ["speaker0_class_label.dat", "speaker1_class_label.dat", "speaker2_class_label.dat", "speaker3_class_label.dat", "speaker4_class_label.dat",
                   "speaker5_class_label.dat", "speaker6_class_label.dat", "speaker7_class_label.dat", "speaker8_class_label.dat", "speaker9_class_label.dat"]
original_speaker_path = '/mnt/alderaan/mlteam3/Assignment2/data/sp/'

output_vaded_path = '/mnt/alderaan/mlteam3/Assignment2/data/speaker_vaded/'

'''



#speaker_list =  ["speakermodel_mfcc.dat","ubm_mfcc.dat","vad_mfcc.dat"]
#classlabel_list =["speakermodel_class_label.dat","ubm_class_label.dat","vad_class_label.dat"]

original_speaker_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/ubm/filewise_mfcc/'

class_label_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/ubm/filewise_class_labels'

output_vaded_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/ubm_vaded/'

all_speaker_names = os.walk(original_speaker_path).next()[1]


def removeZerosFromMFCC(speaker_mfcc_file, classlabel_file, output_file):
    print 'Current File: ' + str(speaker_mfcc_file)
    #MFCC FILE
    with open(speaker_mfcc_file, 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()

    # CLASS LABEL FILE
    with open(classlabel_file, 'rb') as infile2:
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

    mfcc_vaded = open(output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(ones)
    cPickle.dump(temp1, mfcc_vaded, -1)
    mfcc_vaded.close()

for speaker in all_speaker_names:
    tmp = []
    for files in os.walk(os.path.join(original_speaker_path,speaker)):
        tmp = files[2]

    for mfccfile in tmp:
        output_file_path = os.path.join(output_vaded_path, speaker)
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        output_file = os.path.join(output_file_path,mfccfile)

        speaker_mfcc_file = os.path.join(original_speaker_path, speaker, mfccfile)

        class_label_filename = mfccfile.replace("mfcc_", "class_label_")
        class_label_file = os.path.join(class_label_path, speaker, class_label_filename)

        removeZerosFromMFCC(speaker_mfcc_file, class_label_file, output_file)
