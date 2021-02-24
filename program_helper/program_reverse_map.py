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
from program_helper.sequence.reverse_map_field import FieldReverseMapper
from program_helper.surrounding.reverse_map_ret import SurroundingReverseMapper
from synthesis.reverse_map.reverse_map_ast import AstReverseMapper
from program_helper.sequence.reverse_map_fp import FPReverseMapper
from program_helper.set.reverse_map_ret import RetReverseMapper
from utilities.basics import reconstruct_camel_case


class ProgramRevMapper:
    def __init__(self, vocab):
        self.vocab = vocab
        self.ast_mapper = AstReverseMapper(vocab)
        self.fp_mapper = FPReverseMapper(vocab)
        self.field_mapper = FieldReverseMapper(vocab)
        self.ret_mapper = RetReverseMapper(vocab)
        self.surr_mapper = SurroundingReverseMapper(vocab)
        self.api_list = []
        self.type_list = []
        self.keyword_list = []
        self.method_list = []
        self.class_list = []
        self.javadoc_list = []
        self.all_var_mappers = []
        self.return_reached = []
        self.num_data = 0



    def update_data_size(self):
        assert self.ast_mapper.num_data == self.fp_mapper.num_data
        assert self.fp_mapper.num_data == self.ret_mapper.num_data

        self.num_data = self.ast_mapper.num_data
        return

    def add_batched_data(self, loader_batch):
        nodes, edges, targets, var_decl_ids, ret_reached, \
        node_type_number, \
        type_helper_val, expr_type_val, ret_type_val, \
        all_var_mappers, iattrib,\
        ret_type, fp_in, fields, \
        apicalls, types, keywords, method, classname, javadoc_kws, \
        surr_ret, surr_fp, surr_method = loader_batch

        self.add_data(nodes, edges, targets, var_decl_ids, \
                node_type_number, ret_reached,\
                type_helper_val, expr_type_val, ret_type_val, \
                all_var_mappers, \
                ret_type, fp_in, fields, \
                apicalls, types, keywords, method, classname, javadoc_kws,\
                    surr_ret, surr_fp, surr_method)

        return

    def add_data(self, nodes, edges, targets, var_decl_ids, \
                node_type_number, return_reached,\
                type_helper_val, expr_type_val, ret_type_val, \
                all_var_mappers, \
                ret_type, fp_in, fields, \
                apicalls, types, keywords, method, classname, javadoc_kws,\
                    surr_ret, surr_fp, surr_method
                 ):
        self.ast_mapper.add_data(nodes, edges, targets,
                                 var_decl_ids,
                                 node_type_number,
                                 type_helper_val, expr_type_val, ret_type_val
                                 )
        self.fp_mapper.add_data(fp_in)
        self.field_mapper.add_data(fields)
        self.ret_mapper.add_data(ret_type)
        self.update_data_size()
        self.api_list.extend(apicalls)
        self.type_list.extend(types)
        self.keyword_list.extend(keywords)
        self.method_list.extend(method)
        self.class_list.extend(classname)
        self.javadoc_list.extend(javadoc_kws)
        self.surr_mapper.add_data(surr_ret, surr_fp, surr_method)
        self.all_var_mappers.extend(all_var_mappers)
        self.return_reached.extend(return_reached)

    def get_element(self, id):
        return self.ast_mapper.get_element(id), \
               self.fp_mapper.get_element(id), \
               self.field_mapper.get_element(id), \
               self.ret_mapper.get_element(id), \
               self.api_list[id], self.type_list[id], self.keyword_list[id], \
               self.method_list[id], self.class_list[id], self.javadoc_list[id], \
               self.surr_mapper.get_element(id), \
               self.all_var_mappers[id], \
               self.return_reached[id]

    def decode_paths(self, id, partial=True):
        ast, fp, field, ret, apis, types, kws, m, c, javadoc, surr, all_var_mappers, return_reached =\
            self.get_element(id)
        self.ast_mapper.decode_ast_paths(ast, partial=partial)
        print('{} ;; {} ;; {}'.format(all_var_mappers[:10], all_var_mappers[10:20], all_var_mappers[20:]))
        print('--Keywords--')
        print([self.vocab.chars_apiname[a] for a in apis])
        print([self.vocab.chars_typename[t] for t in types])
        print([self.vocab.chars_kw[kw] for kw in kws])
        self.fp_mapper.decode_fp_paths(fp)
        self.field_mapper.decode_fp_paths(field)
        self.ret_mapper.decode_ret(ret)
        print('--Method--')
        print([self.vocab.chars_kw[kw] for kw in m])
        print('--Class--')
        print([self.vocab.chars_kw[kw] for kw in c])
        print('--Javadoc--')
        print([self.vocab.chars_kw[kw] for kw in javadoc])
        print('--Surrounding--')
        self.surr_mapper.decode(surr)
        print('--Return Reached--')
        print(return_reached)

    def get_return_type(self, id):
        ast, fp, field, ret, apis, types, kws, m, c, javadoc, surr, _, _ = self.get_element(id)
        return ret

    def get_return_type_as_name(self, id, vocab=None):
        ret_typ = self.get_return_type(id)
        ret_typ_name = vocab[ret_typ]
        return ret_typ_name

    def get_fp_types(self, id):
        ast, fp, field, ret, apis, types, kws, m, c, javadoc, surr, _, _ = self.get_element(id)
        return fp

    def get_field_types(self, id):
        ast, fp, field, ret, apis, types, kws, m, c, javadoc, surr, _, _ = self.get_element(id)
        return field


    def get_field_type_as_real_names(self, id, vocab=None):
        field_type_vals = self.get_field_types(id)
        field_typ_names = []
        for typ in field_type_vals:
            field_typ_names.append(vocab[typ])
        return field_typ_names

    def get_fp_type_inputs(self, id):
        ast, fp, field, ret, apis, types, kws, m, c, javadoc, surr, _, _ = self.get_element(id)
        return fp

    def get_fp_type_as_real_names(self, id, vocab=None):
        fp_type_vals = self.get_fp_type_inputs(id)
        fp_type_names = []
        for typ in fp_type_vals:
            fp_type_names.append(vocab[typ])
        return fp_type_names


    def get_fp_ret_and_field_names(self, id, vocab=None):
        fp_type_names = self.get_fp_type_as_real_names(id, vocab=vocab)
        ret_type_name = self.get_return_type_as_name(id, vocab=vocab)
        field_type_names = self.get_field_type_as_real_names(id, vocab=vocab)
        return fp_type_names, ret_type_name, field_type_names

    def get_method_name(self, id):
        ast, fp, field, ret, apis, types, kws, m, c, javadoc, surr, _, _ = self.get_element(id)
        return m

    def get_reconstructed_method_name(self, id, vocab=None):
        m_kws = self.get_method_name(id)
        m_kw_names = []
        for kw in m_kws:
            if kw != 0:
                m_kw_names.append(vocab[kw])
        return reconstruct_camel_case(m_kw_names)


    def get_class_name(self, id):
        ast, fp, field, ret, apis, types, kws, m, c, javadoc, surr, _, _ = self.get_element(id)
        return c

    def get_javadoc(self, id):
        ast, fp, field, ret, apis, types, kws, m, c, javadoc, surr, _, _ = self.get_element(id)
        return javadoc

    def get_surrounding(self, id):
        ast, fp, field, ret, apis, types, kws, m, c, javadoc, surr, _, _ = self.get_element(id)
        return surr

    def reset(self):
        self.ast_mapper.reset()
        self.fp_mapper.reset()
        self.field_mapper.reset()
        self.ret_mapper.reset()
        self.surr_mapper.reset()
        self.api_list = []
        self.type_list = []
        self.keyword_list = []
        self.method_list = []
        self.class_list = []
        self.javadoc_list = []
        self.num_data = 0