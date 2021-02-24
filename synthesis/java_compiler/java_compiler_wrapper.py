from jpype import startJVM, shutdownJVM, java, addClassPath, JClass, JInt

startJVM(convertStrings=False)
import jpype.imports
import subprocess


class JavaCompilerWrapper:

    def __init__(self):
        try:
            pwd = '/home/ubuntu/grammar_vae/synthesis/java_compiler/'
            subprocess.run(["javac", pwd + "JavaCompiler.java"])
            # TODO: not an elegant solution, also in self.close()
            subprocess.run(["mv", pwd + "JavaCompiler.class", "./"])
            self.comp = JClass('JavaCompiler')

        except Exception as err:
            print("Exception: {}".format(err))

    def is_assignable_from(self, type1, type2, debug_print=False):
        res = self.comp.checker(type1, type2)
        if debug_print:
            print("Is {} assignable from {} ? :: {}".format(type2, type1, res))
        return res

    def close(self):
        subprocess.run(["rm", "./JavaCompiler.class"])


if __name__ == "__main__":
    jc = JavaCompilerWrapper()
    jc.is_assignable_from("java.util.List", "java.util.ArrayList", debug_print=True)
    jc.is_assignable_from("java.util.List", "java.awt.Rectangle", debug_print=True)
    jc.is_assignable_from("java.util.ArrayList", "java.util.List", debug_print=True)
    jc.is_assignable_from("java.util.ArrayList", "java.util.ArrayList", debug_print=True)
    jc.is_assignable_from("java.nio.Buffer", "java.nio.ByteBuffer", debug_print=True)
    jc.is_assignable_from("java.nio.Buffer", "java.nio.CharBuffer", debug_print=True)
    jc.is_assignable_from("java.awt.Button", "java.awt.Button", debug_print=True)
    jc.is_assignable_from("java.lang.Object", "java.awt.Button", debug_print=True)
    jc.close()
