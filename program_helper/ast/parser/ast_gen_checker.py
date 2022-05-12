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
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np

from data_extraction.data_reader.manipulator.data_add_attribs import has_next_re, next_re, remove_re
from program_helper.ast.ops import DAPICall, DBranch, DLoop, DVarAccess, DSymtabMod, SINGLE_PARENTS, DVarDecl, \
    DAPIInvoke, DExcept, CONTROL_FLOW_NAMES, DClsInit, DVarAssign, DStop, DType, DAPICallMulti, DAPICallSingle, \
    DReturnVar, DVarDeclCls
from program_helper.ast.ops.concepts.DInternalAPICall import DInternalAPICall
from program_helper.ast.parser.ast_checker import AstChecker
from program_helper.ast.parser.ast_exceptions import \
    VoidProgramException, UndeclaredVarException, TypeMismatchException, \
    ConceptMismatchException, RetStmtNotExistException, UnusedVariableException, NullPtrException
from program_helper.ast.parser.ast_traverser import AstTraverser
from utilities.vocab_building_dictionary import DELIM


class AstGenChecker(AstChecker):

    def __init__(self, vocab,
                 max_loop_num=None,
                 max_branching_num=None,
                 max_variables=None,
                 compiler=None,
                 logger=None):

        super().__init__(max_loop_num, max_branching_num, max_variables)
        self.ast_traverser = AstTraverser()
        self.vocab = vocab
        self.logger = logger
        self.java_compiler = compiler
        self.reset_stats()
        return

    def reset_stats(self):
        self.total, self.passed_count, \
        self.void_count, \
        self.undeclared_var_count, self.type_mismatch_count, self.concept_mismatch_count = 0, 0, 0, 0, 0, 0
        self.unused_var_count, self.nullptr_count = 0, 0

        self.passed_input_var_check, self.failed_input_var_check = 0, 0
        self.passed_field_var_check, self.failed_field_var_check = 0, 0

        self.passed_type_check = 0
        self.passed_nullptr_check = 0
        self.passed_type_check_with_blank = 0


        self.failed_type_check = 0
        self.failed_nullptr_check = 0

        self.passed_expr_type_check, self.passed_ret_type_check, \
            self.passed_formal_type_check, self.passed_stmt_type_check = 0, 0, 0, 0
        self.passed_expr_type_check_with_blank, self.passed_ret_type_check_with_blank, \
            self.passed_formal_type_check_with_blank, self.passed_stmt_type_check_with_blank = 0, 0, 0, 0
        self.failed_expr_type_check, self.failed_ret_type_check, \
            self.failed_formal_type_check, self.failed_stmt_type_check = 0, 0, 0, 0

        self.ret_stmt_exists_failed, self.ret_stmt_exists_passed = 0, 0
        self.unused_var_count_failed, self.unused_var_count_passed = 0, 0
        self.obj_init_count_failed, self.obj_init_count_passed = 0, 0
        self.scope_count_failed, self.scope_count_passed = 0, 0
        self.int_method_count_failed, self.int_method_count_passed = 0, 0
        self.nextcheck_failed, self.nextcheck_passed = 0, 0

        self.all_variables_used_failed, self.null_ptrt_exception_thrown = 0, 0

        self.failed_at_distance = defaultdict(int)
        self.failed_at_distance_for_type = defaultdict(int)
        self.failed_at_distance_for_expr_type = defaultdict(int)
        self.failed_at_distance_for_ret_type = defaultdict(int)
        self.failed_at_distance_for_fp_type = defaultdict(int)
        self.failed_at_distance_for_ret_stmt_type = defaultdict(int)
        self.failed_at_distance_for_var = defaultdict(int)
        self.failed_at_distance_for_nullptr = defaultdict(int)

        self.passed_at_length = defaultdict(int)
        self.failed_at_length = defaultdict(int)
        self.failed_at_length_for_type = defaultdict(int)
        self.failed_at_length_for_expr_type = defaultdict(int)
        self.failed_at_length_for_ret_type = defaultdict(int)
        self.failed_at_length_for_fp_type = defaultdict(int)
        self.failed_at_length_for_ret_stmt_type = defaultdict(int)
        self.failed_at_length_for_ret_stmt_exists = defaultdict(int)
        self.failed_at_length_for_var = defaultdict(int)
        self.failed_at_length_for_unused_var = defaultdict(int)
        self.failed_at_length_for_nullptr = defaultdict(int)

        self.num_times_var_accesed = defaultdict(int)
        self.num_times_var_accesed_successfully = defaultdict(int)

        ## liveness properties
        self.declared_variables = list()
        self.used_variables = defaultdict(int)

        # self.declared_variables_in_context = defaultdict(list)
        self.used_variables_in_context = defaultdict(list)

        self.surrounding_methods = dict()
        self.hasnextbool = defaultdict(int)

    def check_generated_progs(self, ast_head, init_symtab=None, return_type=None,
                              update_mode=False):
        if init_symtab is None:
            init_symtab = dict()

        # reset this every time we analyze a prog
        self.declared_variables = list()
        self.used_variables = defaultdict(int)

        # self.declared_variables_in_context = defaultdict(list)

        self.var_decl_context = defaultdict(list)
        self.objects_declared, self.objects_initialized = [], []

        self.used_variables_in_context = defaultdict(list)

        if not update_mode:
            self.check_void_programs(ast_head)

        init_symtab.update({'system_package': 0, 'LITERAL': 0, 'no_return_var': 0})

        self.dynamic_type_var_checker(ast_head, symtab=init_symtab, return_type=return_type, update_mode=update_mode,
                                      context="main")

        return

    def get_fp_symtabs(self, fp_head, symtab=None):
        if symtab is None:
            symtab = dict()

        fp_path = self.ast_traverser.depth_first_search(fp_head)

        j = 0
        for item in fp_path:
            curr_node_val, curr_node_type, curr_node_validity, curr_node_var_decl_ids, \
            parent_node_id, \
            edge_type, expr_type, type_helper, return_type = item
            if curr_node_type == DType.name():
                symtab['fp_' + str(j)] = curr_node_val  # TODO self.vocab[curr_node_val]
                j += 1
        return symtab

    def check_existence_in_symtab(self, id, symtab, name=None, log_location=None, nullptr=False, context=None):

        self.num_times_var_accesed[id] += 1
        self.used_variables[id] += 1

        # self.used_variables_in_context[id] = context
        # Checking for proper variable usage in context
        # id is currently being used in context that we get as input
        # Check for out of context usage
        if 'var_' in id:
            # This checks for declaration check
            usage_context = context
            decl_context = set(list(self.var_decl_context[id]))
            passed = False
            for decl in decl_context:
                if decl in usage_context:
                    passed = True
                    break
            if passed:
                self.scope_count_passed += 1
            else:
                self.scope_count_failed += 1


        # Usage check in context
        self.used_variables_in_context[context].append(id)



        if self.java_compiler.var_violation_check(id, symtab):
            if name is not None:
                self.failure_spot += 'Location :: {}'.format(log_location)
                self.failure_spot += '\nWhile synthesizing API {}'.format(name)
                if not nullptr:
                    self.failure_spot += '\nId {} does not exist in symtab'.format(id)
                else:
                    self.failure_spot += '\nNullptr exception with id {}'.format(id)

            self.failed_at_distance[self.num_statements] += 1

            if not nullptr and 'fp_' in id:
                self.failed_input_var_check += 1
                self.failed_at_distance_for_var[self.num_statements] += 1
                self.var_consistency = False

            elif not nullptr and 'field_' in id:
                self.failed_field_var_check += 1
                self.failed_at_distance_for_var[self.num_statements] += 1
                self.var_consistency = False

            elif nullptr and id not in ['no_return_var']:
                self.failed_nullptr_check += 1
                self.failed_at_distance_for_nullptr[self.num_statements] += 1
                self.nullptr_consistency = False

        else:
            if not nullptr and 'fp_' in id:
                self.passed_input_var_check += 1
            elif not nullptr and 'field_' in id:
                self.passed_field_var_check += 1
            elif nullptr and id not in ['no_return_var']:
                self.passed_nullptr_check += 1

        return

    def check_type_in_symtab(self, id, symtab,
                             ret_type=None,
                             expr_type=None,
                             type_helper=None,
                             log_api_name=None,
                             log_location=None,
                             return_stmt=False,
                             update_mode=False
                             ):

        assert not all([ret_type, expr_type, type_helper]) is None
        if ret_type is not None and self.java_compiler.type_violation_check(id, symtab, ret_type,
                                                                            var_type='ret_type',
                                                                            update_mode=update_mode):
            if log_api_name is not None:
                self.failure_spot += 'Location :: {}'.format(log_location)
                self.failure_spot += '\nWhile synthesizing API {}'.format(log_api_name)
                self.failure_spot += '\nRet Type {} does not match symtab id with type {}:{}\n'.format(ret_type, id,
                                                                                                     symtab[id]
                                                                                                     if id in symtab else 'VarNotExist')
            # raise TypeMismatchException
            self.failed_at_distance[self.num_statements] += 1
            self.failed_at_distance_for_type[self.num_statements] += 1
            self.failed_type_check += 1
            self.type_consistency = False

            if return_stmt is True:
                self.failed_stmt_type_check += 1
                self.failed_at_distance_for_ret_stmt_type[self.num_statements] += 1
                self.ret_stmt_consistency = False
            else:
                self.failed_ret_type_check += 1
                self.failed_at_distance_for_ret_type[self.num_statements] += 1
                self.ret_type_consistency = False

        elif ret_type is not None and return_stmt is True:
            self.passed_stmt_type_check += 1
            self.passed_type_check += 1
            if id in ['no_return_var']:
                self.passed_stmt_type_check_with_blank += 1
                self.passed_type_check_with_blank += 1
            self.num_times_var_accesed_successfully[id] += 1


        elif ret_type is not None:
            self.passed_ret_type_check += 1
            self.passed_type_check += 1
            if id in ['no_return_var']:
                self.passed_ret_type_check_with_blank += 1
                self.passed_type_check_with_blank += 1
            self.num_times_var_accesed_successfully[id] += 1

        if expr_type is not None and self.java_compiler.type_violation_check(id, symtab, expr_type,
                                                                             var_type='expr_type',
                                                                             update_mode=update_mode):
            if log_api_name is not None:
                self.failure_spot += 'Location :: {}'.format(log_location)
                self.failure_spot += '\nWhile synthesizing API {}'.format(log_api_name)
                self.failure_spot += '\nExp Type {} does not match symtab id with type {}:{}\n'.format(expr_type, id,
                                                                                                     symtab[id]
                                                                                                     if id in symtab else 'VarNotExist')
            self.failed_at_distance[self.num_statements] += 1
            self.failed_at_distance_for_type[self.num_statements] += 1
            self.failed_at_distance_for_expr_type[self.num_statements] += 1
            # raise TypeMismatchException
            self.failed_type_check += 1
            self.failed_expr_type_check += 1
            self.type_consistency = False
            self.expr_type_consistency = False

        elif expr_type is not None:
            self.passed_expr_type_check += 1
            self.passed_type_check += 1
            if id in ['system_package', 'LITERAL']:
                self.passed_expr_type_check_with_blank += 1
                self.passed_type_check_with_blank += 1
            self.num_times_var_accesed_successfully[id] += 1

        if type_helper is not None and self.java_compiler.type_violation_check(id, symtab, type_helper,
                                                                               var_type='type_helper',
                                                                               update_mode=update_mode):
            if log_api_name is not None:
                self.failure_spot += 'Location :: {}'.format(log_location)
                self.failure_spot += '\nWhile synthesizing API {}'.format(log_api_name)
                self.failure_spot += '\nType Helper {} does not match symtab id with type {}:{}\n'.format(type_helper, id,
                                                                                                        symtab[id]
                                                                                                        if id in symtab else 'VarNotExist')

            self.failed_at_distance[self.num_statements] += 1
            self.failed_at_distance_for_type[self.num_statements] += 1
            self.failed_at_distance_for_fp_type[self.num_statements] += 1
            # raise TypeMismatchException
            self.failed_type_check += 1
            self.failed_formal_type_check += 1
            self.type_consistency = False
            self.fp_type_consistency = False

        elif type_helper is not None:
            self.passed_formal_type_check += 1
            self.passed_type_check += 1
            if id in ['LITERAL']:
                self.passed_formal_type_check_with_blank += 1
                self.passed_type_check_with_blank += 1
            self.num_times_var_accesed_successfully[id] += 1

        return

    def dynamic_type_var_checker(self, head, symtab=None, update_mode=False, return_type=None, context="main"):

        if head is None:
            return

        if head.type in SINGLE_PARENTS:
            self.dynamic_type_var_checker(head.child,
                                          symtab=symtab,
                                          update_mode=update_mode,
                                          return_type=return_type,
                                          context=context
                                          )
            return

        if symtab is None:
            symtab = {}

        node = head
        while node is not None:
            self.num_statements += 1
            if isinstance(node, DVarDecl):
                num_elems = 0
                for item in symtab.keys():
                    if 'var_' in item:
                        num_elems += 1
                if num_elems < 10:
                    id_ = "var_" + str(self.int_var_mapper_dict[num_elems])
                else:
                    id_ = "var_" + str(num_elems)
                symtab[id_] = node.get_return_type()
                self.declared_variables.append(id_)
                # self.declared_variables_in_context[context].append(id_)
                self.var_decl_context[id_].append(context)
                if isinstance(node, DVarDeclCls):
                    self.objects_declared.append(id_)


            elif isinstance(node, DClsInit):

                singlecallnode = node.get_child()
                ret_id = singlecallnode.sibling.val
                self.check_concept_node(singlecallnode, DAPICallSingle, parent_node_type=DClsInit)
                api_name, _, ret_type = DAPIInvoke.split_api_call(singlecallnode.get_api_call_name())
                # self.check_expr_type(node.get_expr_id(), expr_type, symtab, api_name, update_mode=update_mode)
                self.check_ret_type(node.get_return_id(), ret_type, symtab, api_name,
                                    update_mode=update_mode, nullptr=True, context=context)
                self.check_formal_params(singlecallnode, symtab, update_mode=update_mode,
                                         parent_node_type=DClsInit, context=context)
                self.objects_initialized.append(ret_id)

            elif isinstance(node, DAPIInvoke):
                self.check_api_invoke_node(node, symtab, update_mode=update_mode, context=context)
                self.objects_initialized.append(node.child.sibling.sibling.val)


            elif isinstance(node, DReturnVar):
                ret_id = node.get_ret_id()
                self.check_existence_in_symtab(ret_id, symtab, name="return statement", log_location='ret_id', context=context)
                self.check_type_in_symtab(ret_id, symtab, ret_type=return_type, return_stmt=True)
                self.ret_stmt_exists = True

            elif isinstance(node, DInternalAPICall):
                method_name_id = node.get_return_id() # this is in the form of "local_method_$x"
                method_id = 0
                try:
                    method_id = int(method_name_id.split("_")[-1])
                except:
                    pass
                if method_id in self.surrounding_methods:
                    self.int_method_count_passed += 1
                else:
                    self.int_method_count_failed += 1


            elif node.name() in CONTROL_FLOW_NAMES:
                self.handle_control_flow(node, symtab, update_mode=update_mode, return_type=return_type, context=context)

            elif isinstance(node, DStop):
                pass

            node = node.sibling

        return

    def check_api_invoke_node(self, invoke_node, symtab, update_mode=False, context=None):
        self.check_concept_node(invoke_node.child, DAPICallMulti, parent_node_type=DAPIInvoke)

        singlecallnode = invoke_node.get_first_call_node()
        self.check_concept_node(singlecallnode, DAPICallSingle, parent_node_type=DAPICallMulti)

        api_name, expr_type, ret_type = DAPIInvoke.split_api_call(singlecallnode.get_api_call_name())
        self.check_expr_type(invoke_node.get_expr_id(), expr_type, symtab, api_name, update_mode=update_mode, context=context)
        self.check_ret_type(invoke_node.get_return_id(), ret_type, symtab, api_name, update_mode=update_mode, context=context)

        while not isinstance(singlecallnode, DStop):
            self.check_formal_params(singlecallnode, symtab, update_mode=update_mode, parent_node_type=DAPICallMulti, context=context)
            singlecallnode = singlecallnode.progress_sibling_node()



        # Now lets check for has_next
        if has_next_re.match(api_name):
            self.hasnextbool[context] = 1

        if next_re.match(api_name) and "loop" in context:

            hasnext_exists = False
            if self.hasnextbool[context] == 1:
                hasnext_exists = True
            if context.endswith("then_") or context.endswith("else_") or context.endswith("body_"):
                parent_context = context[:-5]
                if self.hasnextbool[parent_context] == 1:
                    hasnext_exists = True


            if hasnext_exists:
                self.nextcheck_passed += 1
            else:
                self.nextcheck_failed += 1


    def check_concept_node(self, node, node_type, parent_node_type=None):
        if not isinstance(node, node_type):
            self.failure_spot += '{}->{} production rule was violated'.format(parent_node_type, node_type)
            raise ConceptMismatchException

    def check_ret_type(self, ret_id, ret_type, symtab, api_name, update_mode=False, nullptr=True, context=None):
        self.check_existence_in_symtab(ret_id, symtab, name=api_name, log_location='ret_id',
                                       nullptr=nullptr, context=context)
        self.check_type_in_symtab(ret_id, symtab, ret_type=ret_type,
                                  log_api_name=api_name,
                                  log_location='ret id',
                                  update_mode=update_mode
                                  )

    def check_expr_type(self, expr_id, expr_type, symtab, api_name, update_mode=False, context=None):
        self.check_existence_in_symtab(expr_id, symtab, name=api_name,
                                       log_location='expr_id', context=context)
        self.check_type_in_symtab(expr_id, symtab, expr_type=expr_type,
                                  log_api_name=api_name,
                                  log_location='expr id',
                                  update_mode=update_mode
                                  )

    def check_formal_params(self, singlecallnode, symtab, update_mode=False, parent_node_type=None, context=None):

        self.check_concept_node(singlecallnode, DAPICallSingle, parent_node_type=parent_node_type)

        api_call = singlecallnode.get_api_call_name()
        if '(' not in api_call:
            self.failure_spot += 'Bracket missing in API Call'
            raise ConceptMismatchException

        arg_list = DAPICallMulti.get_formal_types_from_data(api_call)
        arg_id = 0

        start = singlecallnode.get_start_of_fp_nodes()
        while start is not None:
            fp_id = start.val
            self.check_existence_in_symtab(fp_id, symtab, name=singlecallnode.get_api_name(),
                                           log_location='formal param id : ' + str(arg_id),
                                           context=context)

            self.check_type_in_symtab(fp_id, symtab, type_helper=arg_list[arg_id],
                                      log_api_name=singlecallnode.get_api_name(),
                                      log_location='formal param id : ' + str(arg_id),
                                      update_mode=update_mode)
            start = start.progress_sibling_node()
            arg_id += 1

    def handle_control_flow(self, node, symtab, update_mode=False, return_type=None, context=None):
        # control flow does not impact the symtab

        hash = str(random.randint(1, 1000))
        if isinstance(node, DBranch):
            temp1 = deepcopy(symtab)
            temp2 = deepcopy(symtab)
            temp3 = deepcopy(symtab)
            self.dynamic_type_var_checker(node.child, symtab=temp1,
                                          update_mode=update_mode,
                                          return_type=return_type,
                                          context=context + ";branch_" + hash
                                          )
            self.dynamic_type_var_checker(node.child.sibling, symtab=temp2,
                                          update_mode=update_mode,
                                          return_type=return_type,
                                          context=context + ";branch_" + hash + "then_"
                                          )
            self.dynamic_type_var_checker(node.child.sibling.sibling, symtab=temp3,
                                          update_mode=update_mode,
                                          return_type=return_type,
                                          context=context + ";branch_" + hash + "else_"
                                          )

        elif isinstance(node, DLoop):
            temp1 = deepcopy(symtab)
            temp2 = deepcopy(symtab)
            self.dynamic_type_var_checker(node.child, symtab=temp1,
                                          update_mode=update_mode,
                                          return_type=return_type,
                                          context=context + ";loop_" + hash
                                          )
            self.dynamic_type_var_checker(node.child.sibling, symtab=temp2,
                                          update_mode=update_mode,
                                          return_type=return_type,
                                          context=context + ";loop_" + hash + "body_"
                                          )

        elif isinstance(node, DExcept):
            temp1 = deepcopy(symtab)
            temp2 = deepcopy(symtab)
            self.dynamic_type_var_checker(node.child, symtab=temp1,
                                          update_mode=update_mode,
                                          return_type=return_type,
                                          context=context + ";except_try_" + hash
                                          )
            self.dynamic_type_var_checker(node.child.sibling, symtab=temp2,
                                          update_mode=update_mode,
                                          return_type=return_type,
                                          context=context + ";except_catch_" + hash
                                          )

        return

    def run_viability_check(self, ast_candies,
                            fp_types=None,
                            ret_type=None,
                            field_vals=None,
                            mapper=None,
                            debug_print=True,
                            surrounding=None
                            ):

        outcome_strings = []
        for ast_candy in ast_candies:
            ast_node = ast_candy.head
            outcome_string = self.check_single_program(ast_node, fp_types=fp_types,
                                                       ret_type=ret_type,
                                                       field_vals=field_vals,
                                                       mapper=mapper,
                                                       surrounding=surrounding
                                                       )

            if debug_print:
                self.logger.info(outcome_string)

            outcome_strings.append(outcome_string)

        return outcome_strings

    def check_single_program(self, ast_node,
                             fp_types=None,
                             ret_type=None,
                             field_vals=None,
                             update_mode=False,
                             mapper=None,
                             surrounding=None
                             ):

        passed = False
        self.failure_spot = ''

        field_mapper = mapper[20:]
        field_map_dict = {i: j for i, j in enumerate(field_mapper)}
        init_symtab = dict()
        for j, val in enumerate(field_vals):
            if val == DELIM:
                continue
            key = 'field_' + str( field_map_dict[j] )
            init_symtab.update({key: val})

        fp_mapper = mapper[10:20]
        fp_map_dict = {i: j for i, j in enumerate(fp_mapper)}
        for j, val in enumerate(fp_types):
            if val == DELIM:
                continue
            key = 'fp_' + str( fp_map_dict[j] )
            init_symtab.update({key: val})

        self.surrounding_methods = dict()
        if surrounding is not None:
            surr_ret, surr_fp, surr_method = surrounding
            for j in range(len(surr_ret)): # number of surrounding methods
                if np.sum(surr_ret[j])>0 or np.sum(surr_fp[j])>0 or np.sum(surr_method[j])>0:
                    self.surrounding_methods[j] = {"ret": surr_ret[j], "fp": surr_fp[j], "name": surr_method[j]}

        self.hasnextbool = defaultdict(int)
        self.objects_declared, self.objects_initialized = [], []

        self.int_var_mapper_dict = {i: j for i, j in enumerate(mapper[:10])}

        try:
            self.var_consistency, self.type_consistency, self.ret_stmt_exists = True, True, False
            self.expr_type_consistency, self.ret_type_consistency, \
            self.fp_type_consistency, self.ret_stmt_consistency = True, True, True, True
            self.nullptr_consistency = True

            self.num_statements = 0
            self.check_generated_progs(ast_node, init_symtab=init_symtab,
                                       return_type=ret_type,
                                       update_mode=update_mode)


            # check if any variable was left unused
            unused_vars = False
            for elem in self.declared_variables:
                if elem not in self.used_variables:
                    self.failed_at_length_for_unused_var[self.num_statements] += 1
                    self.failed_at_length[self.num_statements] += 1
                    unused_vars = True
                    self.unused_var_count_failed += 1
                else:
                    self.unused_var_count_passed += 1


            if self.ret_stmt_exists is False and ret_type not in [DELIM, 'void']:
                self.ret_stmt_exists_failed += 1
            else:
                self.ret_stmt_exists_passed += 1

            objects_initialized = set(self.objects_initialized)
            for var in self.objects_declared:
                # The logic for this "or" logic is to capture cases where
                # objects are initialized with return stmts of api calls
                if var in objects_initialized or var in self.used_variables:
                    if var in objects_initialized:
                        self.obj_init_count_passed += 1
                    else:
                        self.obj_init_count_failed += 1

            fail_reason = ''
            if self.var_consistency is False:
                self.failed_at_length_for_var[self.num_statements] += 1
                self.failed_at_length[self.num_statements] += 1
                raise UndeclaredVarException

            if self.nullptr_consistency is False:
                self.failed_at_distance_for_nullptr[self.num_statements] += 1
                self.failed_at_length[self.num_statements] += 1
                raise NullPtrException

            elif self.type_consistency is False:
                self.failed_at_length_for_type[self.num_statements] += 1
                self.failed_at_length[self.num_statements] += 1
                if self.fp_type_consistency is False:
                    self.failed_at_length_for_fp_type[self.num_statements] += 1
                elif self.expr_type_consistency is False:
                    self.failed_at_length_for_expr_type[self.num_statements] += 1
                elif self.ret_type_consistency is False:
                    self.failed_at_length_for_ret_type[self.num_statements] += 1
                elif self.ret_stmt_consistency is False:
                    self.failed_at_length_for_ret_stmt_type[self.num_statements] += 1
                raise TypeMismatchException

            elif self.ret_stmt_exists is False and ret_type not in [DELIM, 'void']:
                self.failed_at_length_for_ret_stmt_exists[self.num_statements] += 1
                self.failed_at_length[self.num_statements] += 1
                raise RetStmtNotExistException


            if unused_vars:
                raise UnusedVariableException

            self.passed_at_length[self.num_statements] += 1


            self.passed_count += 1
            passed = True
        except VoidProgramException:
            self.void_count += 1 if not update_mode else 0
            fail_reason = 'is void'
        except UndeclaredVarException:
            self.undeclared_var_count += 1 if not update_mode else 0
            fail_reason = 'has undeclared var'
        except TypeMismatchException:
            self.type_mismatch_count += 1 if not update_mode else 0
            fail_reason = 'has mismatched type'
        except RetStmtNotExistException:
            self.ret_stmt_exists_failed += 1 if not update_mode else 0
            fail_reason = 'has no return statement'
        except ConceptMismatchException:
            self.concept_mismatch_count += 1 if not update_mode else 0
            fail_reason = 'has mismatched concept'
        except UnusedVariableException:
            self.unused_var_count += 1 if not update_mode else 0
            fail_reason = 'has unused variable'
        except NullPtrException:
            self.nullptr_count += 1 if not update_mode else 0
            fail_reason = 'has nullptr'

        outcome_string = 'This program passed' if passed else 'This program failed because it ' + fail_reason
        outcome_string += '\n' + self.failure_spot

        self.total += 1

        return outcome_string

    def print_stats(self, logger=None):
        self.logger.info('')
        self.logger.info('{:8d} programs/asts in total'.format(self.total))
        self.logger.info('{:8d} programs/asts missed for concept mismatch'.format(self.concept_mismatch_count))
        self.logger.info('{:8d} programs/asts missed for being void'.format(self.void_count))
        self.logger.info('{:8d} programs/asts missed for illegal var access'.format(self.undeclared_var_count))
        self.logger.info('{:8d} programs/asts missed for type mismatch'.format(self.type_mismatch_count))
        self.logger.info('{:8d} programs/asts missed for return stmt not existing'.format(self.ret_stmt_exists_failed))
        self.logger.info('{:8d} programs/asts missed for unused variables'.format(self.unused_var_count))
        self.logger.info('{:8d} programs/asts missed for nullptr'.format(self.nullptr_count))
        self.logger.info('{:8d} programs/asts passed'.format(self.passed_count))
        self.logger.info('')

        self.print_percentage_stats(passed=self.passed_count+self.void_count,
                                                        passed_blank=self.void_count,
                                                        failed=self.total - self.passed_count-self.void_count,
                                                        info='all progs')


        self.print_percentage_stats(passed=self.scope_count_passed,
                                    passed_blank=0,
                                    failed=self.scope_count_failed,
                                    info='No undeclared variable access')


        self.print_percentage_stats(passed=self.passed_input_var_check+self.passed_nullptr_check,
                                    passed_blank=0,
                                    failed=self.failed_input_var_check+self.failed_nullptr_check,
                                    info='valid formal parameter access')


        self.print_percentage_stats(passed=self.passed_field_var_check,
                                    passed_blank=0,
                                    failed=self.failed_field_var_check,
                                    info='valid class var access')


        self.print_percentage_stats(passed=self.obj_init_count_passed,
                                    passed_blank=0,
                                    failed=self.obj_init_count_failed,
                                    info='no uninitialized objects')

        self.print_percentage_stats(passed=self.scope_count_passed + self.passed_input_var_check
                                           + self.passed_field_var_check + self.passed_nullptr_check,
                                    passed_blank=0,
                                    failed=self.scope_count_failed + self.failed_input_var_check
                                           + self.failed_field_var_check + self.failed_nullptr_check,
                                    info='no variable access error')


        self.print_percentage_stats(passed=self.passed_expr_type_check,
                                    passed_blank=self.passed_expr_type_check_with_blank,
                                    failed=self.failed_expr_type_check,
                                    info='object-method comptibility / expr types')

        self.print_percentage_stats(passed=self.passed_ret_type_check,
                                    passed_blank=self.passed_ret_type_check_with_blank,
                                    failed=self.failed_ret_type_check,
                                    info='ret type at call site')

        self.print_percentage_stats(passed=self.passed_formal_type_check,
                                    passed_blank=self.passed_formal_type_check_with_blank,
                                    failed=self.failed_formal_type_check,
                                    info='actual param types')

        self.print_percentage_stats(passed=self.passed_stmt_type_check,
                                    passed_blank=self.passed_stmt_type_check_with_blank,
                                    failed=self.failed_stmt_type_check,
                                    info='return stmt type')


        self.print_percentage_stats(passed=self.passed_type_check,
                                    passed_blank=self.passed_type_check_with_blank,
                                    failed=self.failed_type_check,
                                    info='No type errors')


        self.print_percentage_stats(passed=self.ret_stmt_exists_passed,
                                    passed_blank=0,
                                    failed=self.ret_stmt_exists_failed,
                                    info='return stmt exists')

        self.print_percentage_stats(passed=self.unused_var_count_passed,
                                    passed_blank=0,
                                    failed=self.unused_var_count_failed,
                                    info='no unused variables')

        self.print_percentage_stats(passed=self.nextcheck_passed,
                                    passed_blank=0,
                                    failed=self.nextcheck_failed,
                                    info='has next safety')

        self.print_percentage_stats(passed=self.int_method_count_passed,
                                    passed_blank=0,
                                    failed=self.int_method_count_failed,
                                    info='uninitiated method access (NOT in paper)')



        # self.logger.info('Percentages at production rule distance')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_distance, total_percentage=100, extra_info='everything')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_distance_for_var, total_percentage=100, extra_info='var_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_distance_for_nullptr, total_percentage=100, extra_info='nullptr_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_distance_for_type, total_percentage=100, extra_info='type_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_distance_for_ret_type, total_percentage=100, extra_info='ret_type_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_distance_for_expr_type, total_percentage=100, extra_info='expr_type_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_distance_for_fp_type, total_percentage=100, extra_info='fp_type_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_distance_for_ret_stmt_type, total_percentage=100, extra_info='ret_stmt_type_checks')

        # self.logger.info('Percentages for program length')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_length, total_percentage=100, extra_info='everything')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_length_for_var, total_percentage=100, extra_info='var_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_length_for_type, total_percentage=100, extra_info='type_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_length_for_ret_type, total_percentage=100, extra_info='ret_type_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_length_for_expr_type, total_percentage=100, extra_info='expr_type_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_length_for_fp_type, total_percentage=100, extra_info='fp_type_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_length_for_ret_stmt_type, total_percentage=100, extra_info='ret_stmt_checks')
        # self.print_pdf_info(fail_stat_dict=self.failed_at_length_for_ret_stmt_exists, total_percentage=100, extra_info='ret_stmt_exists')


        # for i in sorted(self.passed_at_length.keys()):
        #     self.logger.info("Percentage Stat for length {}".format(i))
        #
        #     self.print_percentage_stats(passed=self.passed_at_length[i],
        #                                 passed_blank=0,
        #                                 failed=self.failed_at_length[i],
        #                                 info='all types')

        # self.logger.info('Num times var accessed')
        # self.logger.info(self.num_times_var_accesed)
        #
        # self.logger.info('Num times var accessed successfully')
        # self.logger.info(self.num_times_var_accesed_successfully)

    def print_percentage_stats(self, passed=None, passed_blank=None, failed=None, info=None):
        percentage = passed/(passed + failed + 0.001) * 100.
        percentage_blank = passed_blank/(passed + failed + 0.001) * 100.
        effective_percentage = (passed - passed_blank) / \
                               (passed - passed_blank + failed + 0.001) * 100.

        self.logger.info('  Percentage of {} passed {:0.2f}, {:.2f} was blank, '
                         'effective percentage {:.2f}, count {} '.format(info, percentage, percentage_blank,
                                                                         effective_percentage, passed+failed))
        return

    def print_pdf_info(self, fail_stat_dict, total_percentage, extra_info=''):

        keys = sorted(fail_stat_dict.keys())
        total = 0
        for key in keys:
            total += fail_stat_dict[key]
        self.logger.info('Percentage of fail for {}'.format(extra_info))
        log_info = ''
        for key in keys:
            pdf = fail_stat_dict[key] / total * total_percentage
            log_info += '{:2d}:{:.2f} '.format(key, pdf)
        self.logger.info(log_info)

