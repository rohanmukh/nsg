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
from program_helper.ast.ops import DStop, DBranch, DLoop, DExcept, DVarDecl, \
    DAPIInvoke, DSubTree, DVarAssign, DClsInit, DVarDeclCls
from program_helper.ast.ops.concepts.DAPICallSingle import DAPICallSingle
from program_helper.ast.ops.concepts.DInfix import DInfix
from program_helper.ast.ops.concepts.DInternalAPICall import DInternalAPICall
from program_helper.ast.ops.concepts.DReturnVar import DReturnVar
from utilities.vocab_building_dictionary import DELIM


class JSON_Synthesis:

    def __init__(self):
        return


    def full_json_extract_no_vocab(self, ast_node, name='foo'):
        ast_dict = {}
        _js = self.paths_to_ast(ast_node.head)
        ast_dict['ast'] = {'node': 'DSubTree', '_nodes':  _js}
        ast_dict['method'] = name
        ast_dict['return_type'] = ast_node.return_type
        fps = []
        for fp in ast_node.formal_param_inputs:
            if fp != DELIM:
                fps.append({'_returns': fp})
        ast_dict['formal_params'] = fps
        ast_dict['prob'] = ast_node.log_probability
        return ast_dict


    def full_json_extract(self, ast_node, vocab, name='foo'):
        ast_dict = {}
        _js = self.paths_to_ast(ast_node.head)
        ast_dict['ast'] = {'node': 'DSubTree', '_nodes':  _js}
        ast_dict['method'] = name
        ast_dict['return_type'] = vocab[ast_node.return_type]
        fps = []
        for fp in ast_node.formal_param_inputs:
            fp_ = vocab[fp]
            if fp_ != DELIM:
                fps.append({'_returns': fp_})
        ast_dict['formal_params'] = fps

        fts = []
        for ft in ast_node.field_types:
            ft_ = vocab[ft]
            if ft_ != DELIM:
                fts.append({'_returns': ft_})
        ast_dict['field_types'] = fts

        ast_dict['prob'] = ast_node.log_probability
        return ast_dict

    def paths_to_ast(self, head_node):
        """
        Converts a AST
        :param paths: the set of paths
        :return: the AST
        """
        json_nodes = []
        self.expand_all_siblings_till_STOP(json_nodes, head_node.child)

        return json_nodes

    def expand_all_siblings_till_STOP(self, json_nodes, head_node):
        """
        Updates the given list of AST nodes with those along the path starting from pathidx until STOP is reached.
        If a DBranch, DExcept or DLoop is seen midway when going through the path, recursively updates the respective
        node type.
        :param nodes: the list of AST nodes to update
        :param path: the path
        :param pathidx: index of path at which update should start
        :return: the index at which STOP was encountered if there were no recursive updates, otherwise -1
        """

        while not (isinstance(head_node, DStop) or (head_node is None)):
            node = head_node
            astnode = {}
            if isinstance(node, DBranch):
                astnode['node'] = DBranch.name()
                astnode['_cond'] = []
                astnode['_then'] = []
                astnode['_else'] = []
                self.update_DBranch(astnode, node)
                if len(astnode['_cond']) > 0 or len(astnode['_then']) > 0 or len(astnode['_else']) > 0:
                    json_nodes.append(astnode)
            elif isinstance(node, DLoop):
                astnode['node'] = DLoop.name()
                astnode['_cond'] = []
                astnode['_body'] = []
                self.update_DLoop(astnode, node)
                if len(astnode['_cond']) > 0 or len(astnode['_body']) > 0:
                    json_nodes.append(astnode)
            elif isinstance(node, DExcept):
                astnode['node'] = DExcept.name()
                astnode['_try'] = []
                astnode['_catch'] = []
                self.update_DExcept(astnode, node)
                if len(astnode['_try']) > 0 or len(astnode['_catch']) > 0:
                    json_nodes.append(astnode)

            elif isinstance(node, DVarDecl):
                # DVarDeclCls inherited from here
                json_nodes.append({'node': DVarDecl.name(),
                                   '_returns': node.get_return_type(),
                                   '_id': node.get_synthesized_id()
                                   })

            elif node.val == DAPIInvoke.name():
                astnode['node'] = DAPIInvoke.name()
                astnode['expr_var_id'] = node.child.sibling.val
                astnode['ret_var_id'] = node.child.sibling.sibling.val
                self.update_DAPIInvoke(astnode, node)
                json_nodes.append(astnode)
            elif node.val == DClsInit.name():
                astnode['node'] = DClsInit.name()
                astnode['_id'] = node.child.val
                astnode['_returns'] = node.child.sibling.val
                astnode['_call'] = node.child.sibling.sibling.child.val
                self.update_DClsInit(astnode, node)
                json_nodes.append(astnode)
            elif isinstance(node, DVarAssign):
                json_nodes.append({'node': DVarAssign.name(),
                                   '_returns': node.child.val,
                                   '_id': node.child.sibling.val,
                                   '_rhs_id': node.child.sibling.sibling.sibling.val
                                   })
            elif isinstance(node, DInternalAPICall):

                fps = []
                start = node.child.sibling.sibling
                while start is not None:
                    fps.append(start.val)
                    start = start.sibling

                json_nodes.append({'node': DInternalAPICall.name(),
                                   'ret_var_id': node.child.sibling.val,
                                   'int_method_id': node.child.val,
                                    'fps': fps
                                   })

            elif isinstance(node, DSubTree):
                astnode['node'] = DSubTree.name()
                astnode['_nodes'] = []
                self.expand_all_siblings_till_STOP(astnode['_nodes'], node.child)
                json_nodes.append(astnode)
            elif node.val == DInfix.name():
                astnode['node'] = DInfix.name()
                astnode['_op'] = node.child.val,
                astnode['_left'] = []
                astnode['_right'] = []
                self.update_DInfix(astnode, node)
                json_nodes.append(astnode)
            elif node.val == DReturnVar.name():
                astnode['node'] = DReturnVar.name()
                astnode['_id'] = node.child.val
                json_nodes.append(astnode)
            else:
                astnode['node'] = head_node.val
                json_nodes.append(astnode)

            head_node = head_node.sibling
        return

    def update_DClsInit(self, astnode, cls_node):
        singlecallnode = cls_node.child.sibling.sibling
        start = singlecallnode.child.sibling
        fps = []
        while start is not None:
            fps.append(start.val)
            start = start.sibling
        astnode['fp'] = fps

    def update_DAPIInvoke(self, astnode, invoke_node):
        multicallnode = invoke_node.child
        singlecallnode = multicallnode.child
        astnode['_calls'] = []
        while singlecallnode is not None and singlecallnode.val == DAPICallSingle.name():
            fps = []
            start = singlecallnode.child.sibling
            while start is not None:
                fps.append(start.val)
                start = start.sibling

            curr_node = {
                '_call': singlecallnode.child.val,
                'fp': fps
            }
            astnode['_calls'].append(curr_node)
            singlecallnode = singlecallnode.sibling

    def update_DRetStmt(self, astnode, retstmt_node):
        """
        Updates a DBranch AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_stmt'], retstmt_node.child)
        return


    def update_DInfix(self, astnode, infix_node):
        """
        Updates a DBranch AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_left'], infix_node.child.sibling.child)
        self.expand_all_siblings_till_STOP(astnode['_right'], infix_node.child.sibling.sibling.child)
        return


    def update_DBranch(self, astnode, branch_node):
        """
        Updates a DBranch AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_cond'], branch_node.child.child)
        self.expand_all_siblings_till_STOP(astnode['_then'], branch_node.child.sibling.child)  # then is sibling of if
        self.expand_all_siblings_till_STOP(astnode['_else'],
                                           branch_node.child.sibling.sibling.child)  # else is sibling to if-then
        return

    def update_DExcept(self, astnode, excp_node):
        """
        Updates a DExcept AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_try'], excp_node.child.child)
        self.expand_all_siblings_till_STOP(astnode['_catch'], excp_node.child.sibling.child)
        return

    def update_DLoop(self, astnode, loop_node):
        """
        Updates a DLoop AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """

        self.expand_all_siblings_till_STOP(astnode['_cond'], loop_node.child.child)
        self.expand_all_siblings_till_STOP(astnode['_body'], loop_node.child.sibling.child)
        return

