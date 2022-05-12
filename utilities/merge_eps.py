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
import os


def merge_eps(folder, output_file, regex_input):
    output_file = os.path.join(folder, output_file)
    regex_input = folder + regex_input
    os.system(
        'gs - q - dNOPAUSE - dBATCH - sDEVICE = pdfwrite - dEPSCrop - sOutputFile=' + output_file + ' ' + regex_input)
