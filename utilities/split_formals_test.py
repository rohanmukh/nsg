import argparse
import sys

from program_helper.ast.ops.concepts.DAPICallMulti import DAPICallMulti
from program_helper.ast.ops.concepts.DAPIInvoke import DAPIInvoke


def test_call_types():
    x1 = 'java.lang.ThreadLocal<java.util.Map<java.lang.Class<?>,java.lang.Object>>.' \
         'set(java.util.Map<java.lang.Class<?>,java.lang.Object>)'
    x2 = 'java.lang.String.valueOf(int)'
    x3 = 'java.lang.Integer.valueOf(java.lang.String)'
    x4 = 'java.io.File.length()'
    x5 = 'javax.swing.text.AbstractDocument.AbstractElement.getParent()'
    x6 = 'java.awt.geom.AffineTransform.getTranslateInstance(double,double)'
    x7 = 'java.awt.geom.AffineTransform.getTranslateInstance(Map<double>,Map<double>)'

    for key in [x1, x2, x3, x4, x5, x6, x7]:
        print('Program is {}'.format(key))
        temp = DAPIInvoke.get_expr_type_from_key(key)
        print('Expr is {}'.format(temp))
        temp = DAPICallMulti.get_formal_types_from_data(key)
        print('FPs are {} of length {}'.format(temp, len(temp)))
        print()

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--data', type=str, default='../data_extraction/data_reader/data',
                        help='load data from here')
    clargs_ = parser.parse_args()
    sys.setrecursionlimit(clargs_.python_recursion_limit)

    test_call_types(clargs_)
