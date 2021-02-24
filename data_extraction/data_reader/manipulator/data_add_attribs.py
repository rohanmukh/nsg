import argparse
import re
import sys

import ijson

from utilities.basics import dump_json

has_next_re = re.compile(r'java\.util\.Iterator<.*>\.hasNext()')
next_re = re.compile(r'java\.util\.Iterator<.*>\.next()')
remove_re = re.compile(r'java\.util\.Iterator<.*>\.remove()')


class DataAttributeLabeler:
    def __init__(self, dump_sattrib):
        self.dump_sattrib = dump_sattrib

    def read_programs(self, infile):
        with open(infile, 'rb') as f:
            for program in ijson.items(f, 'programs.item'):
                if 'ast' in program:
                    yield program

    def propagate_attribs_list(self, dnodes, attrib):
        for dnode in dnodes:
            attrib = self.propagate_attribs(dnode, attrib)

        return attrib

    def propagate_attribs(self, dnode, attrib):
        """
        Propagate attrib through the complete ast (dnode)
        Sets the inherited attributes and (optionally) the synthesized attributes.
        Returns the synthesized attributes.
        """
        
        if attrib != [0, 0, 0]:
            dnode['iattrib'] = attrib

        sattrib = None
        node_type = dnode['node']
        if node_type == 'DSubTree':
            sattrib = self.propagate_attribs_list(dnode['_nodes'], attrib)

        elif node_type in ['DVarCall', 'DFieldCall', 'DReturnVar', 'DAssign', 'DExceptionVar']:
            sattrib = attrib

        elif node_type == 'DClsInit':
            sattrib = self.propagate_attribs_list(dnode['fp_var_ids'], attrib)

        elif node_type == 'DInternalAPICall':
            sattrib = self.propagate_attribs_list(dnode['fp_var_ids'], attrib)

        elif node_type == 'DReturnStmt':
            sattrib = self.propagate_attribs_list(dnode['_stmt'], attrib)

        elif node_type == 'DVarDeclExpr':
            sattrib = self.propagate_attribs_list(dnode['_stmt'], attrib)

        elif node_type == 'DAPICall':
            cnodes = [dnode['expr_var_id']] + dnode['fp_var_ids']
            sattrib = self.propagate_attribs_list(cnodes, attrib)

            # Update attributes
            sattrib = sattrib.copy()  # copy before modifying.
            api_name = dnode['_call']
            if has_next_re.match(api_name):
                sattrib = [1, 0, 0]
            elif next_re.match(api_name):
                sattrib[1] = 1
            elif remove_re.match(api_name):
                sattrib[2] = 1

        elif node_type == 'DLoop':
            cnodes = dnode['_cond'] + dnode['_body']

            sattrib = self.propagate_attribs_list(cnodes, attrib)

            # Reset all attribs to initial values after the loop
            sattrib = attrib

        elif node_type == 'DBranch':

            cond_nodes = dnode['_cond']
            then_nodes = dnode['_then']
            else_nodes = dnode['_else']

            sattrib_cond = self.propagate_attribs_list(cond_nodes, attrib)
            sattrib_then = self.propagate_attribs_list(then_nodes, sattrib_cond)
            sattrib_else = self.propagate_attribs_list(else_nodes, sattrib_cond)

            # Currently setting sattrib to be the sattrib of the "then" block if it is non-empty.
            # Ideally if both the blocks are presents, sattrib should be set randomly to one of them.
            if then_nodes:
                sattrib = sattrib_then
            else:
                sattrib = sattrib_else

        elif node_type == 'DExcept':
            try_nodes = dnode['_try']
            catch_nodes = dnode['_catch']
            try_sattrib = self.propagate_attribs_list(try_nodes, attrib)
            catch_sattrib = self.propagate_attribs_list(catch_nodes, attrib)

            # Currently setting sattrib to be the sattrib of the "try" block if it is non-empty.
            if try_nodes:
                sattrib = try_sattrib
            else:
                sattrib = catch_sattrib

        elif node_type == 'DInfix':
            cnodes = [dnode['_left'], dnode['_right']]

            sattrib = self.propagate_attribs_list(cnodes, attrib)

        else:
            raise NotImplemented()

        if self.dump_sattrib:
            dnode['sattrib'] = sattrib

        return sattrib

    def add_attributes(self, program):
        ast = program['ast']

        init_attribs = [0, 0, 0]
        self.propagate_attribs(ast, init_attribs)
        program['ast'] = ast
        return program

    def add_attributes_main(self, infile, outfile):
        """
        Add iterator attributes (hasNext, next, and remove) to each AST node.
        Only inherited attributed are added. (Set dump_sattrib to true if you want
        """
        new_programs = []
        for program in self.read_programs(infile):
            new_program = self.add_attributes(program)
            new_programs.append(new_program)

        dump_json({'programs': new_programs}, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=100000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--input_filename', type=str)
    parser.add_argument('--output_filename', type=str)
    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    d = DataAttributeLabeler(dump_sattrib=False)
    d.add_attributes_main(clargs.input_filename, clargs.output_filename)
