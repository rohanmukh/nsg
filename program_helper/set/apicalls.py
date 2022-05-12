import re


class ApiCalls:
    @staticmethod
    def from_call(callnode):
        call = callnode['_call']
        call = re.sub('^\$.*\$', '', call)  # get rid of predicates
        name = call.split('(')[0].split('.')[-1]
        name = name.split('<')[0]  # remove generics from call name
        return [name] if name[0].islower() else []  # Java convention

