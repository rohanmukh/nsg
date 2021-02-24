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
import java_compiler.dsl.DVarCall;
import java_compiler.dsl.DExceptionVar;
import org.eclipse.jdt.core.dom.*;

public class DOMCatchClause implements Handler {

    final CatchClause clause;
    final Visitor visitor;

    public DOMCatchClause(CatchClause clause, Visitor visitor) {
        this.clause = clause;
        this.visitor = visitor;
    }

    @Override
    public DSubTree handle() {
        
        DSubTree tree = new DSubTree();

        SingleVariableDeclaration svd = clause.getException();
        IVariableBinding ivb = svd.resolveBinding();
        if (ivb != null)
            tree.addNode(new DExceptionVar(ivb));

        DSubTree clause_tree = new DOMBlock(clause.getBody(), visitor).handle();
        
        tree.addNodes(clause_tree.getNodes());
        return tree;
    }
}
