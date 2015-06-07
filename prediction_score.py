import sys, os, shutil
import argparse
import bob
import numpy
import cPickle
import scipy.sparse
import facereclib.utils as utils
from sklearn import preprocessing

scores_dir = '/mnt/alderaan/mlteam3/Assignment2/output/score_dumps'

def getScoreFeatures(score_input):
    # read model features - MFCC Features for the Speaker(s)
    with open(score_input, 'rb') as infile:
        score_features = cPickle.load(infile)
    infile.close()
    return score_features

tmp = []
for files in os.walk(scores_dir):
    tmp = files[2]

for scorefilename in tmp:
#scorefilename = "linear_score_speaker_model_64.score"
    if not "ubm" in scorefilename:
        #print "Evaluating Score File: "+str(scorefilename)
        score_file_path = os.path.join(scores_dir,scorefilename)
        score_features = getScoreFeatures(score_file_path)
        total_speaker_count = int(len(score_features))
        correct_detection = int(0)
        for speaker in score_features:
            maximum_score = -1;
            for key, all_model_scores in speaker.iteritems():
                for indivmodelscore in all_model_scores:
                    for key1,value1 in indivmodelscore.iteritems():
                        if (value1 > maximum_score):
                            maximum_score = value1
            for indivmodelscore in all_model_scores:
                for key2,value2 in indivmodelscore.iteritems():
                    if (value2 == maximum_score):
                        #print str(key)+','+str(key2)
                        if (str(key) == str(key2)):
                            correct_detection+=1
        #print "Correct Detection: "+str(correct_detection)
        #print "Total Number of Speaker Probes: "+str(total_speaker_count)
        #print "Calculating Detection Percentage: "
        perc = float((float(correct_detection) * 100 / float(total_speaker_count)))
        #print "Detection Percent: "+str(perc)+'%'
        print str(scorefilename)+','+str(perc)


