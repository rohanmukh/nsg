import re

from program_helper.ast.ops import DAPICallMulti

OBJECT_TYPE = "java.lang.Object"

def judge_a_typ(typ: str):
    ## Primitive types
    if 'Tau_' in typ:
        return OBJECT_TYPE
    elif '<' in typ:
        return OBJECT_TYPE
    elif '.' not in typ:
        return typ
    elif typ.startswith("java") or typ.startswith("javax"):
        return typ
    else:
        return OBJECT_TYPE

def get_angled_bracket_inside(input_type):
    angle_count = 0
    left = 0
    right = len(input_type)
    for j, s in enumerate(input_type):
        if s == '<' and angle_count == 0:
            left = j+1
        if s == '<':
            angle_count += 1
        if s == '>':
            angle_count -= 1
        if s == '>' and angle_count == 0:
            right = j
    word = input_type[left:right]
    return word, left, right

def separate_words(insider):
    angle_count = 0
    words = []
    curr_arg = ''
    for j, s in enumerate(insider):
        if s == '<':
            angle_count += 1
        elif s == '>':
            angle_count -= 1
        if s == ',' and angle_count == 0:
            if len(curr_arg) > 0:
                words.append(curr_arg)
            curr_arg = ''
        else:
            curr_arg += s
    if len(curr_arg) > 0:
        words.append(curr_arg)
    return words


def simplify_java_types(input_type):
    if input_type is None:
        return OBJECT_TYPE
    if not (input_type.startswith("java") or input_type.startswith("javax")):
        if input_type.startswith("short") or input_type.startswith("int") or input_type.startswith("long") \
                or input_type.startswith("float") or input_type.startswith("double") or input_type.startswith("byte") \
                or input_type.startswith("char") or input_type.startswith("boolean") or input_type == "void":
            return input_type
        else:
            return OBJECT_TYPE

    insider, left, right = get_angled_bracket_inside(input_type)
    words = separate_words(insider)
    new_words = [judge_a_typ(temp) for temp in words]
    return_str = input_type[:left] + ",".join(new_words) + input_type[right:]
    return return_str


def simplify_java_api(input_api):
    if input_api is None:
        print("Erroneous API Call")
        return OBJECT_TYPE
    input_api = input_api.replace('$NOT$', '')

    bracketed_string = re.findall('\([a-zA-Z0-9 ,<>_\[\]\.?@]*\)', input_api)[0]
    words = separate_words(bracketed_string[1:-1])
    new_fp_words = [judge_a_typ(temp) for temp in words]

    other_part = input_api.replace(bracketed_string, '')
    insider, left, right = get_angled_bracket_inside(other_part)
    words = separate_words(insider)
    new_words = [judge_a_typ(temp) for temp in words]
    return_str = other_part[:left] + ",".join(new_words) + other_part[right:]

    # handle bracket
    return_str += '(' + ",".join(new_fp_words) + ')'
    return return_str


if __name__ == "__main__":
    x = "java.util.Map<java.lang.String,java.lang.Integer>"
    print(x)
    y = simplify_java_types(x)
    print(y)
    x = "datenstrukturen.BinBaum<java.lang.Object>"
    print(x)
    y = simplify_java_types(x)
    print(y)
    x = "int"
    print(x)
    y = simplify_java_types(x)
    print(y)
    x = "java.util.List<Tau_E>.iterator()"
    print(x)
    y = simplify_java_types(x)
    print(y)
    x = "java.util.Iterator<java.util.Map.Entry<java.lang.String,org.opencms.gwt.shared.property.CmsClientProperty>, int>.hasNext()"
    print(x)
    y = simplify_java_types(x)
    print(y)
    x = "$NOT$java.util.concurrent.ExecutorService.isTerminated()"
    print(x)
    y = simplify_java_api(x)
    print(y)
    x = "$NOT$java.util.concurrent.ExecutorService.isTerminated(org.asa.awdew)"
    print(x)
    y = simplify_java_api(x)
    print(y)
    # TODO: should None be Object?
    x = None
    print(x)
    y = simplify_java_types(x)
    print(y)