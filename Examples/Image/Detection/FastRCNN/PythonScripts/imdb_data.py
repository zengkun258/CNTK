# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import sys, os
sys.path.append('C:\Users\pabuehle\Desktop\PROJECTS\pythonLibrary')
#from pabuehle_utilities_CV_v1 import *
from cntk_helpers import readGtAnnotation
from pabuehle_utilities_general_v0 import *
import scipy.sparse
import scipy.io as sio
import cPickle
import numpy as np
import fastRCNN
#from fastRCNN import imdb
#import fastRCNN.imdb as imdb
#import fastRCNN
#import imdb as imdb
#import xml.dom.minidom as minidom
#import utils.cython_bbox
#import subprocess
#import imdb
#import fastRCNN.imdb as imdb


class imdb_data(fastRCNN.imdb):
    def __init__(self, image_set, classes, maxNrRois, imgDir, roiDir, cacheDir, boAddGroundTruthRois):
        fastRCNN.imdb.__init__(self, image_set + ".cache") #'data_' + image_set)
        self._image_set = image_set
        self._maxNrRois = maxNrRois
        self._imgDir = imgDir
        self._roiDir = roiDir
        self._cacheDir = cacheDir #cache_path
        self._imgSubdirs ={'train': ['positive', 'negative'], 'test': ['testImages']}
        self._classes = classes
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index, self._image_subdirs = self._load_image_set_index()
        self._roidb_handler = self.selective_search_roidb
        self._boAddGroundTruthRois = boAddGroundTruthRois


    #overwrite parent definition
    @property
    def cache_path(self):
        return self._cacheDir

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_subdirs[i], self._image_index[i])

    def image_path_from_index(self, subdir, fname):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._imgDir, subdir, fname)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Compile list of image indices and the subdirectories they are in.
        """
        image_index = []
        image_subdirs = []
        for subdir in self._imgSubdirs[self._image_set]:
            imgFilenames = getFilesInDirectory(self._imgDir + subdir, self._image_ext)
            image_index += imgFilenames
            image_subdirs += [subdir] * len(imgFilenames)
        return image_index, image_subdirs

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_annotation(i) for i in range(self.num_images)]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb


        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)

        #add ground truth ROIs
        if self._boAddGroundTruthRois:
            roidb = self.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = ss_roidb

        #Keep max of e.g. 2000 rois
        if self._maxNrRois and self._maxNrRois > 0:
            print "Only keeping the first %d ROIs.." % self._maxNrRois
            for i in xrange(self.num_images):
                gt_overlaps = roidb[i]['gt_overlaps']
                gt_overlaps = gt_overlaps.todense()[:self._maxNrRois]
                gt_overlaps = scipy.sparse.csr_matrix(gt_overlaps)
                roidb[i]['gt_overlaps'] = gt_overlaps
                roidb[i]['boxes'] = roidb[i]['boxes'][:self._maxNrRois,:]
                roidb[i]['gt_classes'] = roidb[i]['gt_classes'][:self._maxNrRois]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        # box_list = nrImages x nrBoxes x 4
        box_list = []
        for imgFilename, subdir in zip(self._image_index, self._image_subdirs):
            roiPath = "{}/{}/{}.roi.txt".format(self._roiDir, subdir, imgFilename[:-4])
            assert os.path.exists(roiPath), "Error: rois file not found: " + roiPath
            rois = np.loadtxt(roiPath, np.int32)
            box_list.append(rois)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_annotation(self, imgIndex):
        """
        Load image and bounding boxes info from human annotations.
        """
        #negative images do not have any ground truth annotations
        if self._image_subdirs[imgIndex].lower() == "negative":
            return None

        imgPath = self.image_path_at(imgIndex)
        bboxesPaths = imgPath[:-4] + ".bboxes.tsv"
        labelsPaths = imgPath[:-4] + ".bboxes.labels.tsv"
        assert os.path.exists(bboxesPaths), "Error: ground truth bounding boxes file not found: " + bboxesPaths
        assert os.path.exists(labelsPaths), "Error: ground truth labels file not found: " + bboxesPaths
        bboxes = np.loadtxt(bboxesPaths, np.float32)
        labels = readFile(labelsPaths)

        #remove boxes marked as 'undecided' or 'exclude'
        indicesToKeep = find(labels, lambda x: x!='EXCLUDE' and x!='UNDECIDED')
        bboxes = [bboxes[i] for i in indicesToKeep]
        labels = [labels[i] for i in indicesToKeep]

        # Load object bounding boxes into a data frame.
        num_objs = len(bboxes)
        boxes = np.zeros((num_objs,4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        for bboxIndex,(bbox,label) in enumerate(zip(bboxes,labels)):
            cls = self._class_to_ind[label]
            boxes[bboxIndex, :] = bbox
            gt_classes[bboxIndex] = cls
            overlaps[bboxIndex, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}





#TODO: rename all_boxes to something more meaningful
#TODO: change this function after re-writing 8_eval_map_vocImpl to be simpler
    def evaluate_detections(self, all_boxes, output_dir, boUsePythonImpl=False, use_07_metric=None):
        aps = []
        for cls_ind, cls in enumerate(self._classes):
            if cls != '__background__':
                rec, prec, ap = self._voc_eval(cls_ind, all_boxes) #, 0.5, use_07_metric)
                aps += [ap]
                print('AP for {:>15} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.nanmean(aps)))

#TODO: rename variables
    def _voc_eval(self, cls_ind, all_boxes, ovthresh=0.5, use_07_metric=False):
        """
        Top level function that does the PASCAL VOC evaluation.

        classname: Category name (duh)
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        assert (len(all_boxes) == self.num_classes)
        assert (len(all_boxes[0]) == self.num_images)

        # load ground truth annotations for this class
        # npos = 0
        gtInfo = []
        for imgIndex in range(self.num_images):
            imgPath = self.image_path_at(imgIndex)
            imgSubir  = os.path.normpath(imgPath).split(os.path.sep)[-2]
            if imgSubir != 'negative':
                gtBoxes, gtLabels = readGtAnnotation(imgPath)
                gtBoxes = [box for box, label in zip(gtBoxes, gtLabels) if label == self.classes[cls_ind]]
            else:
                gtBoxes = []
            gtInfo.append({'bbox': np.array(gtBoxes),
                           'difficult': [False] * len(gtBoxes),
                           'det': [False] * len(gtBoxes)})

        # parse detections for this class
        # shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coord+score
        detection_bboxInfo = []
        detection_imageIndices = []
        detection_confidence = []
        for imgIndex in range(self.num_images):
            dets = all_boxes[cls_ind][imgIndex]
            if dets != []:
                for k in xrange(dets.shape[0]):
                    detection_imageIndices.append(imgIndex)
                    detection_confidence.append(dets[k, -1])
                    # the VOCdevkit expects 1-based indices
                    detection_bboxInfo.append([dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1])
        detection_bboxInfo = np.array(detection_bboxInfo)
        detection_confidence = np.array(detection_confidence)

        # compute precision / recall / ap
        # REMOVE npos
        rec, prec, ap = self.voc_computePrecisionRecallAp(
            class_recs=gtInfo,
            confidence=detection_confidence,
            image_ids=detection_imageIndices,
            BB=detection_bboxInfo,
            ovthresh=ovthresh,
            use_07_metric=use_07_metric)
        return rec, prec, ap


    #########################################################################
    # Python evaluation functions (copied from faster-RCNN)
    ##########################################################################
#TODO: change variable names, document, and move to helper functions
    def voc_computePrecisionRecallAp(self, class_recs, confidence, image_ids, BB, ovthresh=0.5, use_07_metric=False):
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        npos = sum([len(cr['bbox']) for cr in class_recs])
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_computeAP(rec, prec, use_07_metric)
        return rec, prec, ap

#TODO: rename and move to helper functions
    def voc_computeAP(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap