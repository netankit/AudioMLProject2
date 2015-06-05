import numpy as np
import time
import sys
import cPickle
import scipy.sparse
import os
import shutil
import math



#speaker_list =  ["speakermodel_mfcc.dat","ubm_mfcc.dat","vad_mfcc.dat"]
#classlabel_list =["speakermodel_class_label.dat","ubm_class_label.dat","vad_class_label.dat"]

original_speaker_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/ubm_vaded/'
output_train_vaded_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/train_vaded/ubm/'
output_test_vaded_path = '/mnt/alderaan/mlteam3/Assignment2/data/filewise_dataset_fixednames/test_vaded/ubm/'

all_speaker_names = os.walk(original_speaker_path).next()[1]

for speaker in all_speaker_names:
    if speaker.startswith('s'):
        tmp = []
        for files in os.walk(os.path.join(original_speaker_path,speaker)):
            tmp = files[2]
        count = int(0)

        count_tmp = int(0)
        num_train_files = int(math.ceil((8 *len(tmp))/10))
        print "Number of Total Files: "+str(len(tmp))
        print "Number of Training Files: "+str(num_train_files)
        for mfccfile in tmp:
            srcfile = os.path.join(original_speaker_path,speaker,mfccfile)
            traindir = os.path.join(output_train_vaded_path,speaker)
            testdir = os.path.join(output_test_vaded_path,speaker)

            if not os.path.exists(traindir):
                    os.makedirs(traindir)
            if not os.path.exists(testdir):
                    os.makedirs(testdir)

            #if (mfccfile.startswith('s')):


            if (count_tmp < num_train_files):
                shutil.copy(srcfile, traindir)
                count_tmp = count_tmp + 1
            else:
                shutil.copy(srcfile, testdir)
            #else:
                '''
                if (count < int(2)):
                    if "_SX" in mfccfile:
                        shutil.copy(srcfile, testdir)
                        count  = count + 1
                    else:
                        shutil.copy(srcfile, traindir)
                else:
                    shutil.copy(srcfile, traindir)
                '''
print "Done!"




