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
from program_helper.ast.ops import DAPICall, DBranch, DLoop, DVarAccess, DSymtabMod, DExcept, DVarAccessDecl
from program_helper.ast.parser.ast_exceptions import TooLongBranchingException, TooLongLoopingException, \
    VoidProgramException, TooManyVariableException, TooManyTryException, TooDeepException


class AstChecker:

    def __init__(self,
                 max_depth=None,
                 max_loop_num=None,
                 max_branching_num=None,
                 max_trys=None,
                 max_variables=None):
        self.max_depth = max_depth
        self.max_loop_num = max_loop_num
        self.max_branching_num = max_branching_num
        self.max_trys = max_trys
        self.max_variables = max_variables
        self.stat = {'try': 0, 'branch': 0,  'loop': 0}
        return

    def check(self, head):
        if self.max_depth is not None:
            self.check_depth(head)

        if self.max_branching_num is not None:
            branch_count = self.check_nested_branch(head)
            if branch_count > 0:
                self.stat['branch'] += 1

        if self.max_loop_num is not None:
            loop_count = self.check_nested_loop(head)
            if loop_count > 0:
                self.stat['loop'] += 1

        if self.max_trys is not None:
            try_count = self.check_nested_exceptions(head)
            if try_count > 0:
                self.stat['try'] += 1

        self.check_void_programs(head)

        self.check_too_many_variables(head)
        return

    def check_nested_branch(self, head):

        if head is None:
            return 0

        count_c = self.check_nested_branch(head.child)  # then
        count_s = self.check_nested_branch(head.sibling)  # else
        local_count = 1 if isinstance(head, DBranch) else 0
        count = local_count + max(count_c, count_s)

        if count > self.max_branching_num:
            raise TooLongBranchingException

        return count

    def check_nested_loop(self, head):
        if head is None:
            return 0

        count_c = self.check_nested_loop(head.child)  # then
        count_s = self.check_nested_loop(head.sibling)  # else
        local_count = 1 if isinstance(head, DLoop) else 0
        count = local_count + max(count_c, count_s)

        if count > self.max_loop_num:
            raise TooLongLoopingException

        return count

    def check_nested_exceptions(self, head):
        if head is None:
            return 0

        count_c = self.check_nested_exceptions(head.child)  # then
        count_s = self.check_nested_exceptions(head.sibling)  # else
        local_count = 1 if isinstance(head, DExcept) else 0
        count = local_count + max(count_c, count_s)

        if count > self.max_trys:
            raise TooManyTryException

        return count


    def check_depth(self, head):
        if head is None:
            return 0
        count = 1
        count += self.check_depth(head.child)
        count += self.check_depth(head.sibling)

        if count > self.max_depth:
            raise TooDeepException

        return count

    def check_void_programs(self, head):
        count_apis = self.count_api_calls(head)
        if count_apis == 0:
            raise VoidProgramException

    def count_api_calls(self, head):
        if head is None:
            return 0

        if head.type == DAPICall.name():
            count = 1
        else:
            count = 0
        return count + \
               self.count_api_calls(head.child) + \
               self.count_api_calls(head.sibling)

    def check_too_many_variables(self, head):
        if head is None:
            return

        if (isinstance(head, DVarAccess) or isinstance(head, DVarAccessDecl)) and head.is_valid():
            if 'var_' in head.val and int(head.val.split('_')[1]) >= self.max_variables:
                raise TooManyVariableException

            if 'fp_' in head.val and int(head.val.split('_')[1]) >= self.max_variables:
                raise TooManyVariableException

            if 'field_' in head.val and int(head.val.split('_')[1]) >= self.max_variables:
                raise TooManyVariableException

        self.check_too_many_variables(head.child)
        self.check_too_many_variables(head.sibling)

    def print_stats(self, logger=None):
        logger.info('')
        logger.info('{:8d} programs/asts have branch'.format(self.stat['branch']))
        logger.info('{:8d} programs/asts have loops'.format(self.stat['loop']))
        logger.info('{:8d} programs/asts have trys'.format(self.stat['try']))
        pass