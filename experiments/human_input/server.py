# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import subprocess

from data_extraction.data_reader.manipulator.data_manipulator import DataManipulator
from experiments.automated_queries.cleanup import cleanUp
from program_helper.infer_model_helper import InferModelHelper

BEAM_WIDTH = 50
SEED = 200


class SynthesisServer:
    def __init__(self):
        self.logdir = "./log"

        cleanUp(logdir=self.logdir, make_json_files=False)
        model_path = '/home/ubuntu/savedSearchModel/'
        self.infer_model_helper = InferModelHelper(model_path=model_path,
                                                   beam_width=BEAM_WIDTH, seed=SEED)
        self.manipulator = DataManipulator(debug_print=False)
        return

    def runDomDriver(self, queryFile, fileOut=None):
        print("Extracting Initial JSON ... ", end="")
        sys.stdout.flush()

        java_jar = "/home/ubuntu/grammar_vae/data_extraction/tool_files/dom_driver/target/" \
                   "dom_driver-1.0-jar-with-dependencies.jar"
        configFile = "/home/ubuntu/grammar_vae/data_extraction/java_compiler/config-full.json"

        subprocess.run(["java", "-jar", java_jar, "-f", queryFile, "-c", configFile, "-o", fileOut])

        print("Done")
        return

    def run_query(self, queryFile=None):
        dom_driver_dump_path = self.logdir + '/out.json'
        if os.path.exists(dom_driver_dump_path):
            os.remove(dom_driver_dump_path)

        self.runDomDriver(queryFile, fileOut=dom_driver_dump_path)
        self.manipulator.read_data(dom_driver_dump_path,
                                   output_file=os.path.join(self.logdir, 'out_manipulated.json'),
                                   repair_mode=False,
                                   )

        self.infer_model_helper.read_and_dump_data(filepath=os.path.join(self.logdir, 'out_manipulated.json'),
                                                   data_path=os.path.join(self.logdir, 'temp_data'),
                                                   min_num_data=self.infer_model_helper.infer_model.config.batch_size,
                                                   repair_mode=False
                                                   )

        self.infer_model_helper.synthesize_programs(data_path=os.path.join(self.logdir, 'temp_data'),
                                                    debug_print=False,
                                                    viability_check=True,
                                                    dump_result_path=self.logdir,
                                                    dump_psis=True,
                                                    dump_jsons=False,
                                                    max_programs=1,
                                                    real_ast_jsons=self.infer_model_helper.reader.
                                                    program_reader.ast_storage_jsons,
                                                    )
        self.infer_model_helper.reset()
        self.manipulator.reset()


if __name__ == "__main__":
    synthesizer = SynthesisServer()
    queryFile = "/home/ubuntu/queries/query.java"
    while True:
        id_ = input('\n\nEnter Query number[0/1/2/3/4..9] :: ')
        try:
            id_ = int(id_)
            if not (0 <= id_ < 100):
                raise Exception
        except:
            print("Please enter a number from 0 to 9!")
            continue

        queryFile_splits = queryFile.split('.')
        temp_queryFile = queryFile_splits[0] + str(id_) + "." + queryFile_splits[1]
        if not os.path.exists(temp_queryFile):
            print("Query File {} does not exist".format(temp_queryFile))
            continue

        synthesizer.run_query(queryFile=temp_queryFile)
