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
from program_helper.ast.search.ast_beam_searcher import TreeBeamSearcher


class ProgramBeamSearcher:

    def __init__(self, infer_model):
        self.infer_model = infer_model
        self.beam_width = infer_model.config.batch_size

        self.tree_beam_searcher = TreeBeamSearcher(infer_model)
        return

    def beam_search_memory(self, initial_state=None,
                           ret_type=None,
                           fp_types=None,
                           field_types=None,
                           surrounding=None,
                           mapper=None,
                           method_embedding=None
                           ):

        ast_candies = self.tree_beam_searcher.beam_search(
            initial_state=initial_state,
            ret_type=ret_type,
            formal_params=fp_types,
            field_types=field_types,
            probs=[0. if j == 0 else None for j in range(self.beam_width)],
            mapper=mapper,
            surrounding=surrounding,
            method_embedding=method_embedding
        )

        return ast_candies

