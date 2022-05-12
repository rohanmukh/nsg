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
from program_helper.ast.ops.concepts.DAPICallMulti import DAPICallMulti
from program_helper.ast.ops.concepts.DAPICallSingle import DAPICallSingle
from program_helper.ast.ops import CHILD_EDGE, Node, DAPIInvoke, DVarDecl, DClsInit, DVarAssign, DBranch, \
    DExcept, DLoop, DStop, DSubTree, DCond, DBody, DThen, DElse, DTry, DSymtabMod, DVarAccess, DType, DAPICall, DCatch, \
    CONTROL_FLOW_NAMES, SIBLING_EDGE, DVarDeclCls
import numpy as np
from copy import deepcopy

from program_helper.ast.ops.concepts.DExceptionVarDecl import DExceptionVarDecl
from program_helper.ast.ops.concepts.DInfix import DInfix, DLeft, DRight
from program_helper.ast.ops.concepts.DInternalAPICall import DInternalAPICall
from program_helper.ast.ops.concepts.DReturnVar import DReturnVar
from program_helper.ast.ops.leaf_ops.DClsType import DClsType
from program_helper.ast.ops.leaf_ops.DInternalMethodAccess import DInternalMethodAccess
from program_helper.ast.ops.leaf_ops.DOp import DOp

CONCEPT_NODE, API_NODE, TYPE_NODE,\
        CLSTYPE_NODE, VAR_NODE, OP_NODE, METHOD_NODE, SYMTAB_MOD = 0, 1, 2, 3, 4, 5, 6, 7

NODETYPE_MAPPERS = { CONCEPT_NODE:'CONCEPT_NODE',  API_NODE:'API_NODE',  TYPE_NODE:'TYPE_NODE',  CLSTYPE_NODE:'CLSTYPE_NODE',
                  VAR_NODE:'VAR_NODE',  OP_NODE:'OP_NODE', METHOD_NODE:'METHOD_NODE',  SYMTAB_MOD:'SYMTAB_MOD',
                  }

class Candidate_AST:
    def __init__(self, initial_state, initial_symtab, ret_type,
                 formal_params,
                 field_types,
                 surrounding,
                 prob, mappers, method_embedding):

        ## SYNTHESIS
        self.head = self.tree_currNode = DSubTree()

        self.curr_node_val = self.head.val
        self.curr_edge = CHILD_EDGE
        self.next_node_type = CONCEPT_NODE
        self.fixed_next_node = None
        self.control_flow_stack = []

        ## NEW HELPERS
        self.return_type = ret_type
        self.formal_param_inputs = formal_params
        self.field_types = field_types
        self.surrounding = surrounding

        ## ATTRIBUTES
        # Though never modified or used
        self.type_helper_val_queue = list()
        self.expr_type_val_queue = list()
        self.ret_type_val_queue = list()

        # VARIABLE ACCESSES
        self.var_decl_id = 0
        self.mappers = mappers

        # RETURN REACHED AND IATTRIB
        self.return_reached = False
        self.iattrib = [False, False, False] # iattrib is a triple of booleans, meaning (hasNext, next, remove ) was used

        self.stop_apisingle = False
        ## DEBUGGING PURPOSE
        self.storage = []

        ## BEAM SEARCH PURPOSES
        self.length = 1
        self.log_probability = -np.inf if prob is None else prob
        self.rolling = True
        self.state = initial_state

        ## SYMTAB
        self.symtab = initial_symtab[0]
        self.init_unused_varflag = initial_symtab[1]
        self.init_nullptr_varflag = initial_symtab[2]

        self.method_embedding = method_embedding

        ## COUNTER
        self.single_api_count = 0

    def is_rolling(self):
        return self.rolling

    def is_not_rolling(self):
        return not self.is_rolling()

    def stop_rolling(self):
        self.rolling = False
        return

    def length_mod_and_check(self, curr_val, max_length):
        self.length += 1 \
            if self.is_rolling() and not self.next_node_type == SYMTAB_MOD \
            else 0
            #\
            # if self.next_node_type == CONCEPT_NODE\
            #    and curr_val in MAJOR_CONCEPTS + CONTROL_FLOW_NAMES  \
            #     else 0
        if self.length >= max_length:
            self.stop_rolling()
        return

    def force_finish(self):
        if self.is_rolling():
            self.tree_currNode.add_node(DStop(), edge=self.curr_edge)
        # Then we will handle the control flow stack
        while len(self.control_flow_stack) != 0:
            element = self.control_flow_stack.pop()
            self.tree_currNode = element.get_current_node()
            if self.tree_currNode.val in CONTROL_FLOW_NAMES:
                for edge in element.get_edge_path()[:-1]:
                    self.tree_currNode = self.tree_currNode.progress_node(edge=edge is CHILD_EDGE)
                self.curr_edge = element.get_edge_path()[-1]
                self.tree_currNode.add_node(DStop(), edge=self.curr_edge)
            else:
                self.tree_currNode.valid = False

        self.head = self.remove_invalid(self.head)
        return

    @staticmethod
    def check_valid(node):
        if node is None:
            return True

        if node.valid is False:
            return False

        return Candidate_AST.check_valid(node.child) and Candidate_AST.check_valid(node.sibling)


    def remove_invalid(self, head):
        if head is None:
            return head

        node = head.child
        last_node = head
        last_edge = CHILD_EDGE
        while node is not None:
            if node.val not in [DAPIInvoke.name(), DClsInit.name(), DVarDecl.name(),
                                DVarDeclCls.name(), DExceptionVarDecl.name(),
                                DStop.name(), DInternalAPICall.name(),
                                DVarAssign.name(),
                                DBranch.name(), DLoop.name(), DExcept.name(),
                                DInfix.name(), DReturnVar.name()]:
                print(node.val)

            valid = node.valid and Candidate_AST.check_valid(node.child)


            ## If invalid make connection with last node
            if valid is False:
                if last_edge is CHILD_EDGE:
                    last_node.child = node.sibling
                else:
                    last_node.sibling = node.sibling
            else:
                ## Update last node
                last_node = node
                last_edge = SIBLING_EDGE

            if node.val == DBranch.name():
                if node is not None:
                    node.child = self.remove_invalid(node.child)
                if node.child is not None:
                    node.child.sibling = self.remove_invalid(node.child.sibling)
                if node.child.sibling is not None:
                    node.child.sibling.sibling = self.remove_invalid(node.child.sibling.sibling)

            if node.val == DLoop.name():
                if node is not None:
                    node.child = self.remove_invalid(node.child)
                if node.child is not None:
                    node.child.sibling = self.remove_invalid(node.child.sibling)

            if node.val == DExcept.name():
                if node is not None:
                    node.child = self.remove_invalid(node.child)
                if node.child is not None:
                    node.child.sibling = self.remove_invalid(node.child.sibling)

            if node.val == DInfix.name():
                if node.child is not None:
                    node.child.sibling = self.remove_invalid(node.child.sibling)
                if node.child.sibling is not None:
                    node.child.sibling.sibling = self.remove_invalid(node.child.sibling.sibling)

            ## anyway node has to progress
            node = node.sibling

        return head



    def add_to_storage(self):

        self.storage.append(
            [#self.curr_node_val,
            self.tree_currNode.val,
            self.truncate(self.log_probability),
            self.length
             # self.curr_edge,
             # self.var_decl_id,
             # deepcopy(self.type_helper_val_queue),
             # deepcopy(self.expr_type_val_queue),
             # deepcopy(self.ret_type_val_queue)
             ])
        return

    def truncate(self, x):
        if x == -np.inf:
            return x
        return float(int(x*100))/100

    def debug_print(self, j, vocab_types):
        print('id is {}'.format(j))
        for i in range(len(self.storage[0])):
            for val in self.storage:
                print(val[i], end=',')
            print()

    def print_symtab(self):
        print( 'Var Decl Id :: ' + str(self.var_decl_id) + ' Symtab sum :: '  + str(np.sum(self.symtab))   )

    def add_node(self, value2add):
        node = self.resolve_node_type(value2add)
        self.tree_currNode = self.tree_currNode.add_and_progress_node(node,
                                                                      edge=self.curr_edge)
        return self

    def resolve_node_type(self, value):
        if value == DAPIInvoke.name():
            # DEBUG this
            node = DAPIInvoke()
        elif value == DAPICallMulti.name():
            node = DAPICallMulti()
        elif value == DAPICallSingle.name():
            node = DAPICallSingle()
        elif value == DVarDecl.name():
            node = DVarDecl()
        elif value == DVarDeclCls.name():
            node = DVarDeclCls()
        elif value == DExceptionVarDecl.name():
            node = DExceptionVarDecl()
        elif value == DReturnVar.name():
            node = DReturnVar()
        elif value == DClsInit.name():
            node = DClsInit()
        elif value == DVarAssign.name():
            node = DVarAssign()
        elif value == DInternalAPICall.name():
            node = DInternalAPICall()
        # split ops
        elif value == DBranch.name():
            node = DBranch()
        elif value == DExcept.name():
            node = DExcept()
        elif value == DLoop.name():
            node = DLoop()
        # Delims
        elif value == DSubTree.name():
            node = DSubTree()
        elif value == DStop.name():
            node = DStop()
        elif value == DCond.name():
            node = DCond()
        elif value == DBody.name():
            node = DBody()
        elif value == DThen.name():
            node = DThen()
        elif value == DElse.name():
            node = DElse()
        elif value == DTry.name():
            node = DTry()
        elif value == DCatch.name():
            node = DCatch()
        elif value == DInfix.name():
            node = DInfix()
        elif value == DLeft.name():
            node = DLeft()
        elif value == DRight.name():
            node = DRight()
        ## base node types
        elif self.next_node_type == API_NODE:
            # value = DAPIInvoke.split_api_call(value)[0]
            node = DAPICall(value)
        elif self.next_node_type == TYPE_NODE:
            node = DType(value)
        elif self.next_node_type == CLSTYPE_NODE:
            node = DClsType(value)
        elif self.next_node_type == VAR_NODE:
            node = DVarAccess(val=value)
        elif self.next_node_type == OP_NODE:
            node = DOp(value)
        elif self.next_node_type == SYMTAB_MOD:
            node = DSymtabMod(val=value)
        elif self.next_node_type == METHOD_NODE:
            node = DInternalMethodAccess(val=value)
        else:
            print(
                "Unknown node generated :: value is " + str(value) + " next node type is  " + str(self.next_node_type))
            node = DStop()
        return node

    def get_ret_expr_helper_types(self):
        ## The ordering of the three pop operations is very important,
        ## since the resulting length of one impacts the other
        temp = self.ret_type_val_queue.pop() \
            if len(self.ret_type_val_queue) != 0 \
               and len(self.type_helper_val_queue) == 0 \
               and len(self.expr_type_val_queue) == 0 \
               and self.next_node_type == VAR_NODE \
            else 0
        ret_type_val = [temp]

        temp = self.expr_type_val_queue.pop() \
            if len(self.expr_type_val_queue) != 0 \
               and len(self.type_helper_val_queue) == 0 \
               and self.next_node_type == VAR_NODE \
            else 0
        expr_type_val = [temp]

        temp = self.type_helper_val_queue.pop() \
            if len(self.type_helper_val_queue) != 0 \
               and self.next_node_type in [VAR_NODE, SYMTAB_MOD] \
            else 0
        type_helper_val = [temp]

        return ret_type_val, expr_type_val, type_helper_val


    def get_return_reached(self):
        return self.return_reached


    def get_iattrib(self):
        return self.iattrib
