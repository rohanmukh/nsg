from itertools import chain

import re

from data_extraction.data_reader.utils import split_camel, STOP_WORDS, gather_calls
from program_helper.set.apicalls import ApiCalls
from program_helper.set.types import Types


class Keywords:
    @staticmethod
    def from_call(callnode):
        call = callnode['_call']
        call = re.sub('^\$.*\$', '', call)  # get rid of predicates
        qualified = call.split('(')[0]
        qualified = re.sub('<.*>', '', qualified).split('.')  # remove generics for keywords

        # add qualified names (java, util, xml, etc.), API calls and types
        keywords = list(chain.from_iterable([split_camel(s) for s in qualified if s not in ['java', 'javax']])) + \
                   list(chain.from_iterable([split_camel(c) for c in ApiCalls.from_call(callnode)])) + \
                   list(chain.from_iterable([split_camel(t) for t in Types.from_call(callnode)]))

        # convert to lower case, omit stop words and take the set
        return list(set([k.lower() for k in keywords if k.lower() not in STOP_WORDS]))


