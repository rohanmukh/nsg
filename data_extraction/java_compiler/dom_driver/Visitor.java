/*
Copyright 2017 Rice University
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package java_compiler.dom_driver;

import com.google.common.collect.Multiset;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java_compiler.dsl.DASTNode;
import java_compiler.dsl.DVarCall;
import java_compiler.dsl.DSubTree;
import java_compiler.dsl.Sequence;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.eclipse.jdt.core.dom.*;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.stream.Collectors;

public class Visitor extends ASTVisitor {

    public final CompilationUnit unit;
    public final Options options;
    private final JSONOutput _js;

    public List<MethodDeclaration> allMethods;
    public List<FieldDeclaration> allTypes;

    // call stack during driver execution
    public final Stack<MethodDeclaration> callStack = new Stack<>();

    class JSONOutput {
        List<JSONOutputWrapper> programs;

        JSONOutput() {
            this.programs = new ArrayList<>();
        }
    }

    class JSONOutputWrapper {

        //Identifiers
        String file;
        String method;
        String body;
        String javaDoc;

        // Output
        DSubTree ast;
        DSubTree field_ast;

        // New Evidences Types
        String returnType;
        List<DVarCall> formalParam;

        String className;

        public JSONOutputWrapper(String methodName, String body, DSubTree ast,  DSubTree field_ast,
        String returnType, List<DVarCall> formalParam, String javaDoc, String className) {

            this.file = options.file;
            this.method = methodName;
            this.body = body;

            this.ast = ast;
            this.field_ast = field_ast;

            this.returnType = returnType;
            this.formalParam = formalParam;
            this.javaDoc = javaDoc;

            this.className = className;

        }
    }

    public Visitor(CompilationUnit unit, Options options) throws FileNotFoundException {
        this.unit = unit;
        this.options = options;

        _js = new JSONOutput();
        allMethods = new ArrayList<>();
        allTypes = new ArrayList<>();
    }

    @Override
    public boolean visit(TypeDeclaration clazz) {
        if (clazz.isInterface())
            return false;
        List<TypeDeclaration> classes = new ArrayList<>();
        classes.addAll(Arrays.asList(clazz.getTypes()));
        classes.add(clazz);

        // synchronized lists
        for (TypeDeclaration cls : classes) {

            // System.out.println(cls);
            allTypes = new ArrayList<>();
            allTypes.addAll(Arrays.asList(cls.getFields()));

            String className = cls.getName().getIdentifier();
            DSubTree field_ast = new DSubTree();

            for (FieldDeclaration f : allTypes){
               for (Object v_temp: f.fragments()){
                  VariableDeclarationFragment v = (VariableDeclarationFragment) v_temp;
                  DSubTree t = new DOMVariableDeclarationFragment(v, this).handle();
                  field_ast.addNodes(t.getNodes());
               }
            }


            List<DSubTree> asts = new ArrayList<>();
            List<String> javaDocs = new ArrayList<>();
            List<String> methodNames = new ArrayList<>();
            List<String> returnTypes = new ArrayList<>();
            List<String> bodys = new ArrayList<>();
            List<List<DVarCall>> formalParams = new ArrayList<>();

            allMethods = new ArrayList<>();
            allMethods.addAll(Arrays.asList(cls.getMethods()));
            Collections.shuffle(allMethods);

            List<MethodDeclaration> constructors = allMethods.stream().filter(m -> m.isConstructor()).collect(Collectors.toList());
            List<MethodDeclaration> publicMethods = allMethods.stream().filter(m -> !m.isConstructor() && Modifier.isPublic(m.getModifiers())).collect(Collectors.toList());

            
           
            if (!constructors.isEmpty()) { // no public methods, only constructor
                for (MethodDeclaration c : constructors) {
                    String javadoc = Utils.getJavadoc(c, options.JAVADOC_TYPE);
                    callStack.push(c);
                    DSubTree ast = new DSubTree();
                    ast.addNodes(new DOMMethodDeclaration(c, this).handle().getNodes());
                    callStack.pop();
                    if (ast.isValid()) {
                        asts.add(ast);
                        javaDocs.add(javadoc);
                        methodNames.add(c.getName().getIdentifier() + "@" + getLineNumber(c));
                        returnTypes.add(getReturnType(c));
                        bodys.add(c.toString());
                        formalParams.add(getFormalParams(c));
                    }
                }
            }
 
            if(!publicMethods.isEmpty()){ 
                for (MethodDeclaration m : publicMethods){
                      String javadoc = Utils.getJavadoc(m, options.JAVADOC_TYPE);
                      callStack.push(m);
                      DSubTree ast = new DSubTree();
                      ast.addNodes(new DOMMethodDeclaration(m, this).handle().getNodes());
                      callStack.pop();
                      if (ast.isValid()) {
                          asts.add(ast);
                          javaDocs.add(javadoc);
                          methodNames.add(m.getName().getIdentifier() + "@" + getLineNumber(m));
                          returnTypes.add(getReturnType(m));
                          bodys.add(m.toString());
                          formalParams.add(getFormalParams(m));
                      }
                }
            }

            for (int i = 0; i < asts.size(); i++) {
                DSubTree ast = asts.get(i);
                String javaDoc = javaDocs.get(i);
                String methodName = methodNames.get(i);
                String returnType = returnTypes.get(i);
                List<DVarCall> formalParam = formalParams.get(i);
                String body = bodys.get(i);

                addToJson(methodName, body, ast, field_ast, returnType, formalParam, javaDoc, className);
            }
        }

        return false;
    }

    private void addToJson(String methodName, String body, DSubTree ast, DSubTree field_ast, String returnType,
    List<DVarCall> formalParam, String javaDoc, String className) {
       JSONOutputWrapper out = new JSONOutputWrapper(methodName, body, ast, field_ast, returnType, formalParam, javaDoc, className);
       _js.programs.add(out);
   }

    public String buildJson() throws IOException {
        if (_js.programs.isEmpty())
            return null;

        Gson gson = new GsonBuilder().setPrettyPrinting().serializeNulls().create();

        return gson.toJson(_js);

    }

    public String getReturnType(MethodDeclaration m){
      String ret = null;
      return ret;
    }

    public List<DVarCall> getFormalParams(MethodDeclaration m){
      ArrayList<DVarCall> params = new ArrayList<DVarCall>();
      for (Object p : m.parameters()) {
          VariableDeclaration v = (VariableDeclaration) p;
          DVarCall c = new DVarCall(v.resolveBinding()) ;
          //variableDeclaration.getStructuralProperty(SingleVariableDeclaration.TYPE_PROPERTY).toString();
          params.add(c);
      }
      return params;
    }

    public int getLineNumber(ASTNode node) {
        return unit.getLineNumber(node.getStartPosition());
    }
}
