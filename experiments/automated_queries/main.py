import random
import subprocess

import sys
import json
import os

from experiments.automated_queries.cleanup import cleanUp
from experiments.automated_queries.processJSON import processJSONs, num_of_exps
from program_helper.infer_model_helper import InferModelHelper
from trainer_vae.utils import read_config
from utilities.basics import conditional_director_creator

'''

Experiment 1 :  only use sorrounding information

Experiment 2 :  sorrounding information + javadoc

Experiment 3 : use sorrounding information + javadoc + RT + FP

Experiment 4 : use sorrounding information + jD + RT + FP +
               10 pc of (apicalls, types, keywords)

Experiment 5 : use all of above + add sequence information
'''

BEAM_WIDTH = 10
SEED = 200


def sampleFiles(queryFilesSampled, k):
    print("Sampling Files .... ", end="")
    sys.stdout.flush()
    # sample 10K random files
    randomFiles = random.sample(list(open('/home/ubuntu/github-java-files/github-java-files-test.txt',
                                          encoding = "ISO-8859-1").read().splitlines()) , k)

    with open(queryFilesSampled, "a") as f:
        for randFile in randomFiles:
            randFile = randFile[2:]
            randFile = '/home/ubuntu/java_projects/' + randFile + '\n'
            f.write(randFile)

    print("Done")
    return

def runBatchDomDriver(queryFilesSampled, queryFilesInJson, logdir):
    print("Extracting Initial JSON ... ", end="")
    sys.stdout.flush()

    fileStdOut = logdir + '/L2stdoutDomDriver.txt'
    fileStdErr = logdir + '/L2stderrDomDriver.txt'

    java_jar = "/home/ubuntu/grammar_vae/data_extraction/tool_files/batch_dom_driver/target/" \
               "batch_dom_driver-1.0-jar-with-dependencies.jar"
    configFile = "/home/ubuntu/grammar_vae/data_extraction/java_compiler/config-full.json"

    with open(fileStdOut, "w") as f1 , open(fileStdErr, "w") as f2:
        subprocess.run(["java" , "-jar",  java_jar , queryFilesSampled, configFile ] , stdout=f1, stderr=f2)
        subprocess.run(["sed" , "-i",  "/^Going/d" , fileStdOut])

    with open(queryFilesInJson, 'w') as f:
        subprocess.run(["sed", "-i",  "s/\//_/g", fileStdOut])
        subprocess.run(["sed", "-i", "s/^/JSONFiles\//g", fileStdOut])
        subprocess.run(["sed",  "s/.java$/.java.json/g", fileStdOut] , stdout=f)

    print("Done")
    return


if __name__ == "__main__":

    logdir = "../log"
    queryFilesSampled = logdir + "/L1SampledQueryFileNamesfiles.txt"
    queryFilesInJson = logdir + '/L3JSONFiles.txt'

    cleanUp(logdir = logdir)
    sampleFiles(queryFilesSampled, k=1000)
    runBatchDomDriver(queryFilesSampled, queryFilesInJson, logdir)

    save_path = '/home/ubuntu/savedSearchModel/'
    count = processJSONs(queryFilesInJson, logdir)


    infer_model = InferModelHelper(model_path=save_path,
                                   beam_width=BEAM_WIDTH,
                                   seed=SEED,
                                   max_num_data=100)

    for expNumber in range(num_of_exps):
         exp_logdir = logdir + "/expNumber_" + str(expNumber)
         result_dir = os.path.join(exp_logdir, 'results')
         conditional_director_creator(result_dir)

         input_jsons = os.path.join(exp_logdir, 'L4TestProgramList.json')
         reader_dump_path = os.path.join(exp_logdir, 'temp_data')
         infer_model.read_and_dump_data(filepath=input_jsons,
                                        data_path=reader_dump_path,
                                        reader_dumps_ast=True)
         with open(os.path.join(reader_dump_path, 'program_asts.json')) as f:
            real_ast_jsons = json.load(f)['programs']
            real_ast_jsons = [{'ast': item} for item in real_ast_jsons]

         infer_model.synthesize_programs(data_path=os.path.join(exp_logdir, 'temp_data'),
                                         debug_print=False,
                                         viability_check=True,
                                         dump_result_path=result_dir,
                                         dump_psis=True,
                                         dump_jsons=False,
                                         real_ast_jsons=real_ast_jsons
                                         )
         infer_model.reset()
         print("Number of programs processed for exp " + str(expNumber) + " is "  + str(count) + "\n\n\n\n\n")
