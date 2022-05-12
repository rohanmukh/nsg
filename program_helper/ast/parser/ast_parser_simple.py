from copy import deepcopy
import re

from program_helper.ast.ops import DSubTree, DStop, DBranch, DExcept, DLoop, Node

class ApiCalls:
    @staticmethod
    def from_call(callnode):
        call = callnode['_call']
        call = re.sub('^\$.*\$', '', call)  # get rid of predicates
        name = call.split('(')[0].split('.')[-1]
        name = name.split('<')[0]  # remove generics from call name
        return [name] if name[0].islower() else [name]  # Java convention


class DAPIInvoke(Node):
    def __init__(self, node_js=None, child=None, sibling=None):
        super().__init__(DAPIInvoke.name(), child, sibling)

        self.type = self.val
        self.val = ApiCalls.from_call(node_js['_calls'][0])[0]
        curr_node = self
        for i in range(len(node_js['_calls'][1:])):
            new_node = Node(ApiCalls.from_call(node_js['_calls'][i])[0])
            self.child = new_node
            curr_node = new_node


    @staticmethod
    def name():
        return 'DAPIInvoke'

    @staticmethod
    def delimiter():
        return "__$$__"

    @staticmethod
    def split_api_call(call):
        '''
        :param call:
        :return: a triple of api name, expr_type and ret_type, in this order
        '''
        vals = call.split(DAPIInvoke.delimiter())
        assert len(vals) == 3
        return vals[0], vals[1], vals[2]


class AstParserSimple:
    def __init__(self):
        self.head = None
        return

    def get_ast_with_memory(self, js, symtab=None):
        '''
        :param js: actual json
        :param symtab: dictionary of AST Nodes from ret type, field and fps
        :return:
        '''
        self.head = self.form_ast(js, symtab=symtab)

        return self.head

    def form_ast(self, js, idx=0, symtab=None):
        if symtab is None:
            symtab = dict()

        i = idx
        head = curr_node = Node()
        while i < len(js):
            new_node = self.create_new_nodes(js[i], symtab)
            # new nodes are only returned if it is not a last_DAPICall node
            # if it is a last_DAPICall node, no new nodes are returned, but
            # already attached to last api node
            curr_node = curr_node.add_and_progress_sibling_node(new_node)
            i += 1

        curr_node.add_sibling_node(Node())

        head.child = head.sibling
        head.sibling = None

        return head

    def create_new_nodes(self, node_js, symtab):
        node_type = node_js['node']

        if node_type == 'DAPIInvoke':
            new_node = DAPIInvoke(node_js, symtab)
        elif node_type == 'DInternalAPICall':
            new_node = Node()
        elif node_type == 'DAPICallSingle':
            new_node = Node()
        elif node_type == 'DVarDecl':
            new_node = Node()
        elif node_type == 'DVarDeclCls':
            new_node = Node()
        elif node_type == 'DExceptionVar':
            new_node = Node()
        elif node_type == 'DClsInit':
            new_node = Node()
        elif node_type == 'DFieldCall':
            new_node = Node()
        elif node_type == 'DReturnVar':
            new_node = Node()
        # Split operation node types
        elif node_type == 'DBranch':
            new_node, symtab = self.read_DBranch(node_js, symtab)
        elif node_type == 'DExcept':
            new_node, symtab = self.read_DExcept(node_js, symtab)
        elif node_type == 'DLoop':
            new_node, symtab = self.read_DLoop(node_js, symtab)
        # elif node_type == 'DInfix':
        #     new_node = self.read_DInfix(node_js, symtab)
        # Else throw exception
        else:
            new_node = Node()

        return new_node


    def read_DLoop(self, js_branch, symtab):
        old_symtab = deepcopy(symtab)
        # assert len(pC) <= 1
        nodeC = self.form_ast(js_branch['_cond'], symtab=symtab)  # will have at most 1 "path"
        nodeC.val = 'DCond'
        nodeB = self.form_ast(js_branch['_body'], symtab=symtab)
        nodeB.val = 'DBody'
        nodeC.add_sibling_node(nodeB)
        return DLoop(child=nodeC), old_symtab

    def read_DExcept(self, js_branch, symtab):
        old_symtab = deepcopy(symtab)
        nodeT = self.form_ast(js_branch['_try'], symtab=symtab)
        nodeT.val = 'DTry'
        nodeC = self.form_ast(js_branch['_catch'], symtab=symtab)
        nodeC.val = 'DCatch'

        nodeT.add_sibling_node(nodeC)
        return DExcept(child=nodeT), old_symtab

    def read_DBranch(self, js_branch, symtab):
        old_symtab = deepcopy(symtab)
        nodeC = self.form_ast(js_branch['_cond'], symtab=symtab)  # will have at most 1 "path"
        nodeC.val = 'DCond'
        freeze_symtab = deepcopy(symtab)
        # assert len(pC) <= 1
        nodeT = self.form_ast(js_branch['_then'], symtab=symtab)
        nodeT.val = 'DThen'
        nodeE = self.form_ast(js_branch['_else'], symtab=freeze_symtab)
        nodeE.val = 'DElse'
        nodeT.add_sibling_node(nodeE)
        nodeC.add_sibling_node(nodeT)
        return DBranch(child=nodeC), old_symtab



def get_paths(t):
    children = []
    if t.sibling:
        children.append(t.sibling)
    if t.child:
        children.append(t.child)

    if len(children) > 0:
        if t.val is not None:
            return tuple((t.val,) + path
                         for child in children
                         for path in get_paths(child))
        else:
            return tuple(path
                         for child in children
                         for path in get_paths(child))

    else:
        if t.val is not None:
            return ((t.val,),)
        else:
            return ((),)

def get_paths_of_calls(t):
    children = []
    if t.sibling:
        children.append(t.sibling)
    if t.child:
        children.append(t.child)

    if len(children) > 0:
        if t.val is None or t.val in ['DExcept', 'DTry', 'DCatch', 'DBranch', 'DCond', 'DThen', 'DElse', 'DLoop']:
            return tuple(path
                         for child in children
                         for path in get_paths_of_calls(child))
        else:
            return tuple((t.val,) + path
                         for child in children
                         for path in get_paths_of_calls(child))

    else:
        if t.val is not None:
            return ((t.val,),)
        else:
            return ((),)