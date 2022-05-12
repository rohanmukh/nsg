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
from collections import defaultdict

import os

import json

from utilities.basics import dump_json, dictOfset2list


class JavaCompiler:

    def __init__(self):
        self.expr_type_can_invoke = defaultdict(list)
        self.ret_type_can_take = defaultdict(list)
        self.fp_type_can_take = defaultdict(list)
        return

    def var_violation_check(self, id, symtab):

        if id not in symtab:
            return True

        return False

    def type_violation_check(self, id, symtab, usage_type, var_type=None,
                             update_mode=False):

        assert var_type in ['expr_type', 'ret_type', 'type_helper']
        if var_type == 'expr_type' and id in ['system_package', 'LITERAL']:
            return False
        if var_type == 'ret_type' and id in ['no_return_var']:
            return False
        if var_type == 'type_helper' and id in ['LITERAL']:
            return False

        ## TODO: only use type violation with var violation before
        if id not in symtab:
            return True

        declared_var_type = symtab[id]

        if update_mode:
            if var_type == 'expr_type':
                if usage_type not in self.expr_type_can_invoke[declared_var_type]:
                    self.expr_type_can_invoke[declared_var_type].append(usage_type)

            if var_type == 'ret_type':
                if usage_type not in self.ret_type_can_take[declared_var_type]:
                    self.ret_type_can_take[declared_var_type].append(usage_type)

            if var_type == 'type_helper':
                if usage_type not in self.fp_type_can_take[declared_var_type]:
                    self.fp_type_can_take[declared_var_type].append(usage_type)
            return False

        else:
            if var_type == 'expr_type':
                if declared_var_type not in self.expr_type_can_invoke:
                    return True

                if usage_type not in self.expr_type_can_invoke[declared_var_type]:
                    return True

            if var_type == 'ret_type':
                if declared_var_type not in self.ret_type_can_take:
                    return True

                if usage_type not in self.ret_type_can_take[declared_var_type]:
                    return True

            if var_type == 'type_helper':
                if declared_var_type not in self.fp_type_can_take:
                    return True

                if usage_type not in self.fp_type_can_take[declared_var_type]:
                    return True

        return False

    def save(self, data_path):
        self.expr_type_can_invoke = self.expr_type_can_invoke
        self.ret_type_can_take = self.ret_type_can_take
        self.fp_type_can_take = self.fp_type_can_take

        js_data = {'expr_type_can_invoke': self.expr_type_can_invoke,
                   'ret_type_can_take': self.ret_type_can_take,
                   'fp_type_can_take': self.fp_type_can_take
                   }
        dump_json(js_data, os.path.join(data_path, 'compiler_data.json'))

    def load(self, data_path):
        with open(os.path.join(data_path, 'compiler_data.json')) as f:
            js_data = json.load(f)
        self.expr_type_can_invoke = js_data['expr_type_can_invoke']
        self.ret_type_can_take = js_data['ret_type_can_take']
        self.fp_type_can_take = js_data['fp_type_can_take']
        return
