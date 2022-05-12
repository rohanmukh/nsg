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

from program_helper.ast.ops import DVarAccess, Node
from utilities.vocab_building_dictionary import DELIM


class DVarAssign(Node):
    def __init__(self, node_js=None, symtab=None, child=None, sibling=None):

        super().__init__(DVarAssign.name(), child, sibling)

        if node_js is None:
            return

        self.my_key = node_js['_from'] if '_from' in node_js else None
        var_type = node_js['_type']
        self.child = DVarAccess(self.my_key, type_helper=var_type)

        self.rhs_key = node_js['_to'] if '_to' in node_js else None

        if self.rhs_key in symtab:
            self.ref_var = symtab[self.rhs_key]
            self.child.sibling = DVarAccess(self.rhs_key, type_helper=var_type)
        else:
            self.child.sibling = DVarAccess(DELIM, type_helper=DELIM)
        self.child.sibling.valid = False

        self.type = self.val

    @staticmethod
    def name():
        return 'DVarAssign'

    def get_rhs_var_id(self):
        return self.child.sibling.val

    def get_return_id(self):
        return self.child.val

    def get_return_type(self):
        return None


