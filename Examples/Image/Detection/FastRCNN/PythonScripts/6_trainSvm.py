#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Train post-hoc SVMs using the algorithm and hyper-parameters from
traditional R-CNN.
"""

import importlib
from fastRCNN.train_svms import SVMTrainer
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)



#################################################
# Parameters
#################################################
experimentName = "exp1"

#no need to change these
cntkParsedOutputDir = cntkFilesDir + "train_parsed/"




#################################################
# Main
#################################################
print "   svm_targetNorm = " + str(svm_targetNorm)
print "   svm_retrainLimit = " + str(svm_retrainLimit)
print "   svm_posWeight = " + str(svm_posWeight)
print "   svm_C = " + str(svm_C)
print "   svm_B = " + str(svm_B)
print "   svm_penality = " + str(svm_penality)
print "   svm_loss = " + str(svm_loss)
print "   svm_evictThreshold = " + str(svm_evictThreshold)
print "   svm_nrEpochs = " + str(svm_nrEpochs)

#init
assert classifier == 'svm', "Error: classifier variable not set to 'svm' but to '{}'".format(classifier)
makeDirectory(trainedSvmDir)
np.random.seed(svm_rngSeed)
imdb = imdbs["train"]
net = DummyNet(4096, imdb.num_classes, cntkParsedOutputDir)
svmWeightsPath, svmBiasPath, svmFeatScalePath = getSvmModelPaths(trainedSvmDir, experimentName)

# add ROIs which significantly overlap with a ground truth object as positives
positivesGtOverlapThreshold = cntk_posOverlapThres['train']
if positivesGtOverlapThreshold and positivesGtOverlapThreshold > 0:
    print "Adding ROIs with gt overlap >= %2.2f as positives ..." % (positivesGtOverlapThreshold)
    existingPosCounter, addedPosCounter = imdbUpdateRoisWithHighGtOverlap(imdb, positivesGtOverlapThreshold)
    print "   Number of positives originally: {}".format(existingPosCounter)
    print "   Added {} ROIs with overlap >= {} as positives (nrImages = {})".format(addedPosCounter, positivesGtOverlapThreshold, imdb.num_images)

# start training
svm = SVMTrainer(net, imdb, im_detect, svmWeightsPath, svmBiasPath, svmFeatScalePath,
                 svm_C, svm_B, svm_nrEpochs, svm_retrainLimit, svm_evictThreshold, svm_posWeight,
                 svm_targetNorm, svm_penality, svm_loss, svm_rngSeed)
svm.train()
print "DONE."
