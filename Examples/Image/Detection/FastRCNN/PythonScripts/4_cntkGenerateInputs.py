import os, sys, importlib
import shutil, time
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)



####################################
# Parameters
####################################
image_sets = ["train", "test"]



####################################
# Main
####################################
#clear imdb cache and other files
if os.path.exists(cntkFilesDir):
    assert(cntkFilesDir.endswith("cntkFiles/"))
    userInput = raw_input('--> INPUT: Press "y" to delete directory ' + cntkFilesDir + ": ")
    if userInput.lower() not in ['y', 'yes']:
        print "User input is %s: exiting now." % userInput
        exit(-1)
    shutil.rmtree(cntkFilesDir)
    time.sleep(0.1) #avoid access problems


#create cntk representation for each image
for image_set in image_sets:
    imdb = imdbs[image_set]
    print "Number of images in set {} = {}".format(image_set, imdb.num_images)
    makeDirectory(cntkFilesDir)

    #open files for writing
    cntkImgsPath, cntkRoiCoordsPath, cntkRoiLabelsPath, nrRoisPath = getCntkInputPaths(cntkFilesDir, image_set)
    with open(nrRoisPath, 'w')        as nrRoisFile, \
         open(cntkImgsPath, 'w')      as cntkImgsFile, \
         open(cntkRoiCoordsPath, 'w') as cntkRoiCoordsFile, \
         open(cntkRoiLabelsPath, 'w') as cntkRoiLabelsFile:

            # for each image, transform rois etc to cntk format
            for imgIndex in range(0, imdb.num_images):
                if imgIndex % 50 == 0:
                    print "Processing image set '{}', image {} of {}".format(image_set, imgIndex, imdb.num_images)
                currBoxes = imdb.roidb[imgIndex]['boxes']
                currGtOverlaps = imdb.roidb[imgIndex]['gt_overlaps']
                imgPath = imdb.image_path_at(imgIndex)
                imgWidth, imgHeight = imWidthHeight(imgPath)

                #  all rois need to be scaled + padded to cntk input image size
                targetw, targeth, w_offset, h_offset, scale = roiTransformPadScaleParams(imgWidth, imgHeight,
                                                                           cntk_padWidth, cntk_padHeight)
                boxesStr = ""
                labelsStr = ""
                nrBoxes = len(currBoxes)
                for boxIndex, box in enumerate(currBoxes):
                    rect = roiTransformPadScale(box, w_offset, h_offset, scale)
                    boxesStr += getCntkRoiCoordsLine(rect, cntk_padWidth, cntk_padHeight)
                    labelsStr += getCntkRoiLabelsLine(currGtOverlaps[boxIndex, :].toarray()[0],
                                                   cntk_posOverlapThres[image_set],
                                                   nrClasses)

                # if less than e.g. 2000 rois per image, then fill in the rest using 'zero-padding'.
                boxesStr, labelsStr = cntkPadInputs(nrBoxes, cntk_nrRois, nrClasses, boxesStr, labelsStr)

                #update cntk data
                nrRoisFile.write("{}\n".format(nrBoxes))
                cntkImgsFile.write("{}\t{}\t0\n".format(imgIndex, imgPath))
                cntkRoiCoordsFile.write("{} |rois{}\n".format(imgIndex, boxesStr))
                cntkRoiLabelsFile.write("{} |roiLabels{}\n".format(imgIndex, labelsStr))
print "DONE."
