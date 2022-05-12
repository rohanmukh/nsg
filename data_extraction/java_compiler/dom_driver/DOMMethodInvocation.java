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
import org.eclipse.jdt.core.dom.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Stack;
import java.util.Set;
import java.util.HashSet;

public class DOMMethodInvocation implements Handler {

    final MethodInvocation invocation;
    final Visitor visitor;
    static Set<String> type_names;

    static
    {
    type_names = new HashSet<String>();
    type_names.add("boolean[]");
    type_names.add("byte[]");
    type_names.add("char[]");
    type_names.add("short[]");
    type_names.add("int[]");
    type_names.add("long[]");
    type_names.add("float[]");
    type_names.add("double[]");
    type_names.add("T[]");
    }    

    public DOMMethodInvocation(MethodInvocation invocation, Visitor visitor) {
        this.invocation = invocation;
        this.visitor = visitor;
    }

    public static DASTNode argumentToASTNode(Expression e, Visitor visitor){
//         System.out.println(e);
        DASTNode var = new DVarCall("LITERAL");
        if ( e == null) {
            var = new DVarCall("LITERAL");
        } else if (e instanceof FieldAccess){
            //System.out.println("It is field access");
            IVariableBinding ivb = ((FieldAccess) e).resolveFieldBinding();
            var = new DVarCall(ivb);
        }else if (e instanceof Name){
            //System.out.println("It is name");
            Name n = (Name) e;
            if (n.resolveBinding() instanceof IVariableBinding && n.isSimpleName()){
                //System.out.println("It is simple name");
                IVariableBinding ivb = (IVariableBinding)(n.resolveBinding());
                var = new DVarCall(ivb);
            }
            else if (n.resolveBinding() instanceof IVariableBinding && n.isQualifiedName()){
                //System.out.println("It is qualified name");
                Name parent_n = ((QualifiedName) n).getQualifier();
                if (parent_n.resolveBinding() instanceof IVariableBinding){
                      IVariableBinding ivb = (IVariableBinding)(n.resolveBinding());
                      IVariableBinding ivb_parent = (IVariableBinding)(parent_n.resolveBinding());
                      DVarCall node = new DVarCall(ivb);
                      DVarCall parent_node = new DVarCall(ivb_parent);

                      //System.out.println(parent_node.get_type_value());
                      if (type_names.contains(parent_node.get_type_value()))
                          var = new DAPICall(node, parent_node); 
                      else
                          var = new DVarDeclExpr(new DVarCall(ivb));
                }
            }
        } else if (e instanceof ArrayAccess){
            //System.out.println("It is array access");
            Expression x = ((ArrayAccess) e).getArray();
            var = argumentToASTNode(x, visitor);
        } else if (e instanceof SuperFieldAccess){
            //System.out.println("It is super field access");
            IVariableBinding ivb = ((SuperFieldAccess) e).resolveFieldBinding();
            var = new DVarCall(ivb);
        } else if (e instanceof Expression){
            //System.out.println("It is expression");
            DSubTree Tchild = new DOMExpression((Expression) e, visitor).handle();
            if (Tchild.isValid()){
                //System.out.println("It is valid expression");
                var = new DVarDeclExpr(Tchild.getNodes());
            }
            else if (e instanceof ParenthesizedExpression){
                var = expressionToASTNode( ((ParenthesizedExpression) e).getExpression(), visitor);
            }
        }
        return var;
    }


    public static DASTNode expressionToASTNode(Expression e, Visitor visitor){
        DASTNode var = new DVarCall("LITERAL");
        //System.out.println(e);
        if ( e == null) {
            //System.out.println("It is null");
            var = new DVarCall("system_package");
        } else if (e instanceof FieldAccess){
            //System.out.println("It is field access");
            IVariableBinding ivb = ((FieldAccess) e).resolveFieldBinding();
            var = new DVarCall(ivb);
        }else if (e instanceof Name){
            //System.out.println("It is name");
            Name n = (Name) e;
            if (n.resolveBinding() instanceof IVariableBinding && n.isSimpleName()){
                //System.out.println("It is simple name");
                IVariableBinding ivb = (IVariableBinding)(n.resolveBinding());
                var = new DVarCall(ivb);
            }
            else if (n.resolveBinding() instanceof IVariableBinding && n.isQualifiedName()){
                //System.out.println("It is qualified name");
                Name parent_n = ((QualifiedName) n).getQualifier();
                // Note that parent_n is not being used
                if (parent_n.resolveBinding() instanceof IVariableBinding){
                      IVariableBinding ivb = (IVariableBinding)(n.resolveBinding());
                      var = new DVarDeclExpr(new DVarCall(ivb));
                }
                else if (parent_n.resolveBinding() instanceof ITypeBinding){
                      //System.out.println("parent name is type");
                      var = new DVarCall("system_package");
                }
            }
            else if (n.resolveBinding() instanceof ITypeBinding){
                //System.out.println("expression is name but of system type");
                var = new DVarCall("system_package");
            }
        } else if (e instanceof ThisExpression){
            //System.out.println("It is method");
            var = new DVarCall("system_package");
        } else if (e instanceof MethodInvocation){
            //System.out.println("It is method");
            var = new DVarCall("last_DAPICall");
        } else if (e instanceof ArrayAccess){
            //System.out.println("It is array access");
            Expression x = ((ArrayAccess) e).getArray();
            var = expressionToASTNode(x, visitor);
        } else if (e instanceof SuperFieldAccess){
            //System.out.println("It is super field access");
            IVariableBinding ivb = ((SuperFieldAccess) e).resolveFieldBinding();
            var = new DVarCall(ivb);
        } else if (e instanceof StringLiteral || e instanceof BooleanLiteral || e instanceof CharacterLiteral || e instanceof NullLiteral || e instanceof NumberLiteral || e instanceof TypeLiteral){
            //System.out.println("It is literal");
            var = new DVarCall("LITERAL");
        } else if (e instanceof Expression){
            //System.out.println("It is expression");
            DSubTree Tchild = new DOMExpression((Expression) e, visitor).handle();
            if (Tchild.isValid()){
                //System.out.println("It is valid expression");
                var = new DVarDeclExpr(Tchild.getNodes());
            }
            else if (e instanceof ParenthesizedExpression){
                var = expressionToASTNode( ((ParenthesizedExpression) e).getExpression(), visitor);
            }
        }
        return var;
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();

        ASTNode p = invocation.getParent();
        IVariableBinding ret_var_ivb = null;
        if (p instanceof Assignment){
            Expression lhs = ((Assignment) p).getLeftHandSide();
            if (lhs instanceof Name){
                IBinding binding = ((Name) lhs).resolveBinding();
                if (binding instanceof IVariableBinding)
                   ret_var_ivb = (IVariableBinding) binding;
            }
        }
        else if (p instanceof VariableDeclaration){
             ret_var_ivb = ( (VariableDeclaration)  p).resolveBinding();
        }
        // add the expression's subtree (e.g: foo(..).bar() should handle foo(..) first)
        DSubTree Texp = new DOMExpression(invocation.getExpression(), visitor).handle();
        tree.addNodes(Texp.getNodes());

        List<DASTNode> fp_var_ids = new ArrayList<DASTNode>();
        DASTNode expr_var_id = new DVarCall("LITERAL");
         
        // evaluate arguments first
        for (Object o : invocation.arguments()) {
            fp_var_ids.add(argumentToASTNode((Expression) o, visitor));
        }

        IMethodBinding binding = invocation.resolveMethodBinding();
        Expression e = invocation.getExpression();
        expr_var_id = expressionToASTNode(e, visitor);

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
        else if (Utils.isRelevantCall(binding) )   {
            try {
                DVarCall _var = new DVarCall(ret_var_ivb);
                tree.addNode(new DAPICall(binding, fp_var_ids, _var, expr_var_id));     
            } catch (DAPICall.InvalidAPICallException exp) {
                // continue without adding the node
            }
        }
        return tree;
    }
}
