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

from program_helper.ast.ops import Node
from utilities.vocab_building_dictionary import DELIM


class DVarAccess(Node):
    def __init__(self, val=0,
                 expr_type=None,
                 return_type=None,
                 type_helper=None,
                 child=None, sibling=None):
        super().__init__(val, child, sibling)

        self.expr_type = expr_type if expr_type is not None else DELIM
        self.return_type = return_type if return_type is not None else DELIM
        self.type_helper = type_helper if type_helper is not None else DELIM
        self.type = DVarAccess.name()

        return


    @staticmethod
    def name():
        return 'DVarAccess'
