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
import json
from copy import deepcopy

from data_extraction.data_reader.manipulator.dataread_exceptions import lastAPIException
from program_helper.ast.ops import DAPIInvoke, DAPICallMulti
from utilities.process_java_type import simplify_java_types, simplify_java_api, OBJECT_TYPE
from utilities.vocab_building_dictionary import DELIM
from trainer_vae.utils import read_config


def check_is_primitive(input_type):
    if input_type.startswith("short") or input_type.startswith("int") or input_type.startswith("long") \
            or input_type.startswith("float") or input_type.startswith("double") or input_type.startswith("byte") \
            or input_type.startswith("char") or input_type.startswith("boolean") or input_type == "void":
        return True
    return False

class DataPreProcessor:

    def __init__(self):
        with open("/home/ubuntu/save_ultimate_decoderagain/config.json") as f:
            config = read_config(json.load(f), infer=True)
            self.api_dict_keys = list(config.vocab.api_dict.keys())

    def find_clsinit(self, type):
        # TODO: remove this

        for item in self.api_dict_keys:
            if item == DELIM:
                continue
            api_call, expr_type, ret_type = DAPIInvoke.split_api_call(item)
            if expr_type != DELIM:
                continue

            # bracket = re.findall('\([a-zA-Z0-9 ,<>_\[\]\.?@]*\)', api_call)[0]
            # api_call_name = api_call.replace(bracket, '')

            if api_call.startswith(type):
                fps = DAPICallMulti.get_formal_types_from_data(api_call)
                fp_types = [None for i in range(len(fps))]
                return api_call, fp_types

        return None, None

    def full_preprocess(self, js, repair_mode=False, symtab=None):
        self.my_decl_var_id = 0
        new_js = self.preprocess(js, symtab=symtab)

        if repair_mode:
            if 'my_return_var' not in symtab or symtab['my_return_var'] is None:
                symtab['my_return_var'] = 'void'
                node_js_varcall = {'node': 'DReturnVar', '_id': None, '_returns': 'void'}
                if 'iattrib' in new_js[-1]:
                    node_js_varcall['iattrib'] = new_js[-1]['iattrib']
                new_js.append(node_js_varcall)
        else:
            symtab['my_return_var'] = None

        return new_js

    def handle_DVarDeclExpr(self, node, symtab=None, extra_js_nodes=None):
        # id = node["_id"]
        return_type = simplify_java_types(node['_stmt'][-1]['_returns'])

        node_js_varcall = {'node': 'DVarDeclCls',
                           '_id': 'my_decl_var_' + str(self.my_decl_var_id),
                           '_returns': return_type}
        if 'iattrib' in node:
            node_js_varcall['iattrib'] = node['iattrib']

        self.my_decl_var_id += 1
        symtab[node_js_varcall['_id']] = node_js_varcall['_returns']
        # First node add
        extra_js_nodes.append(node_js_varcall)


        node['_stmt'][-1]['ret_var_id'] = node_js_varcall['_id']
        node['_stmt'][-1]['_id'] = None

        node['_stmt'] = self.preprocess(node['_stmt'], symtab=symtab)
        # Second node add
        extra_js_nodes.extend(node['_stmt'])
        processed_id = node_js_varcall['_id']
        return processed_id


    def handle_DAPICallasExp(self, node, symtab=None, extra_js_nodes=None):
        # id = node["_id"]
        return_type = simplify_java_types(node['_returns'])

        node_js_varcall = {'node': 'DVarDecl', '_id': 'my_decl_var_' + str(self.my_decl_var_id),
                           '_returns': return_type}
        if 'iattrib' in node:
            node_js_varcall['iattrib'] = node['iattrib']
        self.my_decl_var_id += 1
        symtab[node_js_varcall['_id']] = node_js_varcall['_returns']
        # First node add
        extra_js_nodes.append(node_js_varcall)
        node['ret_var_id'] = node_js_varcall['_id']

        # Second node add
        node['_calls'] = []
        first_call = {'_call': node['_call']+'()', 'fp': node['fp_var_ids'], '_returns': node['_returns']}
        del node['fp_var_ids'], node['_call']
        node['_calls'].append(first_call)
        node['expr_var_id'] = node['expr_var_id']['_id']
        node['node'] = 'DAPIInvoke'

        extra_js_nodes.append(node)
        processed_id = node_js_varcall['_id']
        return processed_id


    def process_exp_var(self, node, symtab):
        processed_expr_id = None
        extra_js_nodes = []
        node_type = node["node"]
        if node_type == 'DVarCall':
            processed_expr_id = node["_id"]
        elif node_type == 'DVarDeclExpr':
            if '_returns' in node['_stmt'][-1]:
                processed_expr_id = self.handle_DVarDeclExpr(node, symtab=symtab, extra_js_nodes=extra_js_nodes)
        elif node_type == 'DAPICall':
            processed_expr_id = self.handle_DAPICallasExp(node, symtab=symtab, extra_js_nodes=extra_js_nodes)
        else:
            print("Unknown node {} encountered in method arguments".format(node_type))
        return processed_expr_id, extra_js_nodes

    def process_fp_vars(self, fp_vars, symtab):
        processed_fp_ids = []
        extra_js_nodes = []

        for node in fp_vars:
            node_type = node["node"]
            if node_type == 'DVarCall':
                processed_fp_ids.append(node["_id"])
            elif node_type == 'DVarDeclExpr':
                if '_returns' in node['_stmt'][-1]:
                    processed_fp_id = self.handle_DVarDeclExpr(node, symtab=symtab, extra_js_nodes=extra_js_nodes)
                    processed_fp_ids.append(processed_fp_id)
            elif node_type == 'DAPICall':
                processed_fp_id = self.handle_DAPICallasExp(node, symtab=symtab, extra_js_nodes=extra_js_nodes)
                processed_fp_ids.append(processed_fp_id)
            else:
                print("Unknown node {} encountered in method arguments".format(node_type))

        return processed_fp_ids, extra_js_nodes

    def preprocess(self, js, symtab=None, ihvcatch=False):

        if symtab is None:
            symtab = dict()

        new_js = []

        i = 0
        while i < len(js):
            node = js[i]
            if "_returns" in node:
                node["_returns"] = simplify_java_types(node["_returns"])

            if node["node"] == "DAPICall":
                node["_call"] = simplify_java_api(node["_call"])

                processed_fp_ids, extra_js_nodes = self.process_fp_vars(node["fp_var_ids"], symtab)
                node['fp_var_ids'] = processed_fp_ids
                new_js.extend(extra_js_nodes)


                processed_expr_id, extra_js_nodes = self.process_exp_var(node["expr_var_id"], symtab)
                node['expr_var_id'] = processed_expr_id
                expr_id = node['expr_var_id']
                new_js.extend(extra_js_nodes)


                ret_id = node['ret_var_id']


                if expr_id == 'last_DAPICall':
                    try:
                        last_node = new_js[-1]
                        assert last_node['node'] == 'DAPIInvoke'
                    except:
                        raise lastAPIException
                    if 'nested_api' not in last_node:
                        last_node['nested_api'] = []
                    next_call = {'_call': node['_call'], 'fp': node['fp_var_ids'], '_returns': node['_returns']}
                    last_node['_calls'].append(next_call)

                elif ret_id is not None and \
                        (ret_id not in symtab or
                         symtab[ret_id] != node['_returns']):
                    node_js_varcall = {'node': 'DVarDecl', '_id': node['ret_var_id'],
                                       '_returns': node['_returns']}
                    if 'iattrib' in node:
                        node_js_varcall['iattrib'] = node['iattrib']
                    new_js.append(node_js_varcall)
                    # The api node
                    node['_calls'] = []
                    first_call = {'_call': node['_call'], 'fp': node['fp_var_ids'], '_returns': node['_returns']}
                    del node['fp_var_ids'], node['_call']
                    node['_calls'].append(first_call)
                    node['node'] = 'DAPIInvoke'
                    new_js.append(node)
                else:
                    # The api node
                    node['_calls'] = []
                    first_call = {'_call': node['_call'], 'fp': node['fp_var_ids'], '_returns': node['_returns']}
                    del node['fp_var_ids'], node['_call']
                    node['_calls'].append(first_call)
                    node['node'] = 'DAPIInvoke'
                    new_js.append(node)

            elif node["node"] == "DVarCall":
                symtab[node['_id']] = node['_returns']
                node['node'] = 'DVarDecl'

                if not check_is_primitive(node['_returns']) and \
                        (i + 1) < len(js) and \
                        js[i + 1]['node'] != 'DClsInit':

                    call, fps = self.find_clsinit(node['_returns'])
                    if call is not None:
                        clsinit_node =  {'node': 'DClsInit',
                                         '_id': node['_id'],
                                         '_returns': node['_returns'],
                                        '_call': call, 'fp': fps
                          }
                        if 'iattrib' in node:
                            clsinit_node['iattrib'] = node['iattrib']

                        node['node'] = 'DVarDeclCls'
                        new_js.append(node)
                        new_js.append(clsinit_node)
                    else:
                        new_js.append(node)
                else:
                    new_js.append(node)



            elif node["node"] == "DExceptionVar":
                symtab[node['_id']] = node['_returns']
                new_js.append(node)
                if i + 1 < len(js) and js[i + 1]['node'] == 'DAPICall' and js[i + 1]['expr_var_id'] == node['_id']:
                    pass
                elif ihvcatch:
                    use_excpt_node = {'node': 'DAPIInvoke', 'expr_var_id': node['_id'], '_returns': 'void',
                                      '_calls': [{'fp': [], '_call': 'java.lang.Throwable.printStackTrace()'}],
                                      "ret_var_id": None, '_throws': [],
                                      }
                    if 'iattrib' in node:
                        use_excpt_node['iattrib'] = node['iattrib']
                    new_js.append(use_excpt_node)

            elif node["node"] == "DFieldCall":
                symtab[node['_id']] = node['_returns']
                new_js.append(node)

            elif node["node"] == "DReturnVar":
                node['_returns'] = None if node['_returns'] == "null" else node['_returns']
                symtab["my_return_var"] = node['_returns']
                new_js.append(node)

            elif node["node"] == "DClsInit":
                node["_call"] = simplify_java_api(node["_call"])
                processed_fp_ids, extra_js_nodes = self.process_fp_vars(node["fp_var_ids"], symtab)
                node['fp_var_ids'] = processed_fp_ids
                new_js.extend(extra_js_nodes)

                ## First we deal with the var declaration
                if node['_var'] is not None and \
                        (node['_var'] not in symtab or
                         symtab[node['_var']] != node['_returns']):
                    node_js_varcall = {'node': 'DVarDeclCls', '_id': node['_var'], '_returns': node['_returns']
                                       }
                    if 'iattrib' in node:
                        node_js_varcall['iattrib'] = node['iattrib']

                    symtab[node['_var']] = node_js_varcall['_returns']
                    new_js.append(node_js_varcall)
                elif node['_var'] is not None:
                    for js_node in new_js[::-1]:
                        if js_node["node"] == "DVarDecl" and js_node["_id"] == node['_var']:
                            js_node["node"] = "DVarDeclCls"

                node['_id'] = node['_var']
                node['fp'] = node['fp_var_ids']
                del node['_var']
                del node['fp_var_ids']
                new_js.append(node)

            elif node["node"] == "DReturnStmt":
                if '_returns' in node['_stmt'][-1]:
                    return_type = simplify_java_types(node['_stmt'][-1]['_returns'])
                    node_js_varcall = {'node': 'DVarDecl', '_id': 'my_return_var',
                                       '_returns': return_type}

                    if 'iattrib' in node:
                        node_js_varcall['iattrib'] = node['iattrib']

                    symtab['my_return_var'] = node_js_varcall['_returns']
                    node['_stmt'] = self.preprocess(node['_stmt'], symtab=symtab)
                    node['_stmt'][-1]['ret_var_id'] = 'my_return_var'

                    if node['_stmt'][-1]["node"] == "DClsInit":
                        node_js_varcall["node"] = "DVarDeclCls"

                    new_js.append(node_js_varcall)

                    new_js.extend(node['_stmt'])
                    node_js_varcall = {'node': 'DReturnVar', '_id': 'my_return_var',
                                       '_returns': symtab['my_return_var']}

                    if 'iattrib' in node:
                        node_js_varcall['iattrib'] = node['iattrib']

                    new_js.append(node_js_varcall)

            elif node["node"] == "DBranch":
                new_symtab = deepcopy(symtab)
                node['_cond'] = self.preprocess(node['_cond'], symtab=new_symtab)
                node['_then'] = self.preprocess(node['_then'], symtab=new_symtab)
                node['_else'] = self.preprocess(node['_else'], symtab=new_symtab)
                new_js.append(node)

            elif node["node"] == "DLoop":
                new_symtab = deepcopy(symtab)
                node['_cond'] = self.preprocess(node['_cond'], symtab=new_symtab)
                node['_body'] = self.preprocess(node['_body'], symtab=new_symtab)
                new_js.append(node)

            elif node["node"] == "DExcept":
                new_symtab = deepcopy(symtab)
                node['_try'] = self.preprocess(node['_try'], symtab=new_symtab)
                node['_catch'] = self.preprocess(node['_catch'], symtab=new_symtab, ihvcatch=True)
                new_js.append(node)

            elif node["node"] == "DInfix":
                node['_left'] = self.preprocess(node['_left'], symtab=symtab)
                node['_right'] = self.preprocess(node['_right'], symtab=symtab)
                new_js.append(node)

            elif node["node"] == "DInternalAPICall":
                processed_fp_ids, extra_js_nodes = self.process_fp_vars(node["fp_var_ids"], symtab)
                node['fp_var_ids'] = processed_fp_ids
                new_js.append(node)

            elif node["node"] == "DAssign":
                pass

            else:
                print("Internal API Call invoked during preprocessing :: " + str(node["node"]))
                raise Exception("Node type unknown")

            i += 1
        return new_js
