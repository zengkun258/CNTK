import os, sys, importlib
import shutil, time
import subprocess
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)



####################################
# Parameters
####################################
#cntkCmdStrPattern = "{0}/cntk.exe configFile={1}config.cntk currentDirectory={1}"
cntkCmdStrPattern = "cntk.exe configFile={0}configbs.cntk currentDirectory={0} "
#cntkCmdStrPattern = "{}/run.bat"


####################################
# Main
####################################
#copy template files
shutil.copy(cntkTemplateDir + "configbs.cntk", cntkFilesDir)

#run cntk
tstart = datetime.datetime.now()
os.environ['ACML_FMA'] = str(0)
#cmdStr = cntkCmdStrPattern.format(cntkBinariesDir, cntkFilesDir)
cmdStr = cntkCmdStrPattern.format(cntkFilesDir)
print cmdStr
pid = subprocess.Popen(cmdStr, cwd = cntkFilesDir) #, creationflags=subprocess.CREATE_NEW_CONSOLE)
pid.wait()
print ("Time running cntk [s]: " + str((datetime.datetime.now() - tstart).total_seconds()))

#delete model files written during cntk training
filenames = getFilesInDirectory(cntkFilesDir, postfix = None)
for filename in filenames:
    if filename.startswith('AlexNet.addedRoiLayer'):
        os.remove(cntkFilesDir + filename)
assert pid.returncode == 0, "ERROR: cntk ended with exit code {}".format(pid.returncode)


#parse cntk output
image_sets = ["test", "train"]
for image_set in image_sets:
    print "Parsing CNTK output for image set: " + image_set
    cntkImgsListPath = cntkFilesDir + image_set + ".txt"
    outParsedDir = cntkFilesDir + image_set + "_parsed/"
    if classifier == 'svm':
        cntkOutputPath = cntkFilesDir + image_set + ".h2.y"
    elif classifier == 'nn':
        cntkOutputPath = cntkFilesDir + image_set + ".OutputNodes.z"
    else:
        error

    #write cntk output for each image to separate file
    makeDirectory(outParsedDir)
    parseCntkOutput(cntkImgsListPath, cntkOutputPath, outParsedDir, cntk_nrRois, cntk_featureDimensions[classifier],
                    saveCompressed = True, skipCheck = True) #, skip5Mod = 0)

    #delete cntk output file which can be very large
    #deleteFile(cntkOutputPath)

print "DONE."