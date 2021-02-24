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


import java_compiler.dsl.*;
import java_compiler.dom_driver.DOMMethodInvocation;
import org.eclipse.jdt.core.dom.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Stack;


public class DOMClassInstanceCreation implements Handler {

    final ClassInstanceCreation creation;
    final Visitor visitor;

    public DOMClassInstanceCreation(ClassInstanceCreation creation, Visitor visitor) {
        this.creation = creation;
        this.visitor = visitor;
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();
        // add the expression's subtree (e.g: foo(..).bar() should handle foo(..) first)

        ASTNode p = creation.getParent();
        IVariableBinding ret_var_ivb = null;
        String expr_var_id = new String("");
        if (p instanceof Assignment){
            Expression lhs = ((Assignment) p).getLeftHandSide();
            if (lhs instanceof Name){
               IBinding binding = ((Name) lhs).resolveBinding();
               if (binding instanceof IVariableBinding)
                   ret_var_ivb = (IVariableBinding) binding;
            }
        }
        else if (p instanceof VariableDeclaration)
             ret_var_ivb = ( (VariableDeclaration)  p).resolveBinding();

        DSubTree Texp = new DOMExpression(creation.getExpression(), visitor).handle();
        tree.addNodes(Texp.getNodes());

        List<DASTNode> fp_var_ids = new ArrayList<DASTNode>();

        // evaluate arguments first
        for (Object o : creation.arguments()) {
            fp_var_ids.add(DOMMethodInvocation.argumentToASTNode((Expression) o, visitor));
        }

        IMethodBinding binding = creation.resolveConstructorBinding();

        // check if the binding is of a generic type that involves user-defined types
        if (binding != null) {
            ITypeBinding cls = binding.getDeclaringClass();
            boolean userType = false;
            if (cls != null && cls.isParameterizedType())
                for (int i = 0; i < cls.getTypeArguments().length; i++){
                    userType |= !cls.getTypeArguments()[i].getQualifiedName().startsWith("java.")
                            && !cls.getTypeArguments()[i].getQualifiedName().startsWith("javax.");
                }


            if (userType || cls == null) // get to the generic declaration
                while (binding != null && binding.getMethodDeclaration() != binding)
                    binding = binding.getMethodDeclaration();
        }

        MethodDeclaration localMethod = Utils.checkAndGetLocalMethod(binding, visitor);
        if (localMethod != null) {
            String local_id = localMethod.getName().getIdentifier() + "@" + visitor.getLineNumber(localMethod); //Utils.getMethodId(binding, visitor);
            Stack<MethodDeclaration> callStack = visitor.callStack;
            if (! callStack.contains(localMethod)) {
                tree.addNode(new DInternalAPICall(local_id, fp_var_ids, new DVarCall(ret_var_ivb)));
            }
        }
        else if (Utils.isRelevantCall(binding)) {
            try {
                tree.addNode(new DClsInit(new DAPICall(binding, fp_var_ids), new DVarCall(ret_var_ivb)));
            } catch (DAPICall.InvalidAPICallException exp) {
                // continue without adding the node
            }
        }
        return tree;
    }
}
