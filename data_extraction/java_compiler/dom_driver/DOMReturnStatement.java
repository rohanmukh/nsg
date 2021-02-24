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

import java_compiler.dsl.DSubTree;
import java_compiler.dsl.DReturnVar;
import java_compiler.dsl.DReturnStmt;
import org.eclipse.jdt.core.dom.*;


public class DOMReturnStatement implements Handler {

    final ReturnStatement statement;
    final Visitor visitor;

    public DOMReturnStatement(ReturnStatement statement, Visitor visitor) {
        this.statement = statement;
        this.visitor = visitor;
    }

    @Override
    public DSubTree handle() {

        DSubTree tree = new DSubTree();
        DReturnVar temp = null; 
        IVariableBinding ivb = null;
        Expression e = statement.getExpression();
        if(e instanceof Name){
             IBinding binding = ((Name) e).resolveBinding();
             if (binding instanceof IVariableBinding)
                ivb = (IVariableBinding) binding;
                 temp = new DReturnVar(ivb);
             tree.addNode(temp);
        }
        else{
            DSubTree _nodes = new DOMExpression(statement.getExpression(), visitor).handle();
            if (_nodes.isValid())
               tree.addNode(new DReturnStmt(_nodes.getNodes()));
        }
        

        return tree;

        
    }
}
