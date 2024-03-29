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

import java_compiler.dsl.DExcept;
import java_compiler.dsl.DSubTree;
import org.eclipse.jdt.core.dom.CatchClause;
import org.eclipse.jdt.core.dom.TryStatement;

public class DOMTryStatement implements Handler {

    final TryStatement statement;
    final Visitor visitor;

    public DOMTryStatement(TryStatement statement, Visitor visitor) {
        this.statement = statement;
        this.visitor = visitor;
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();

        // restriction: considering only the first catch clause
        DSubTree Ttry = new DOMBlock(statement.getBody(), visitor).handle();
        DSubTree Tcatch;
        if (! statement.catchClauses().isEmpty())
            Tcatch = new DOMCatchClause((CatchClause) statement.catchClauses().get(0), visitor).handle();
        else
            Tcatch = new DSubTree();
        DSubTree Tfinally = new DOMBlock(statement.getFinally(), visitor).handle();
        
        Tcatch.addNodes(Tfinally.getNodes());
        boolean except = Ttry.isValid() && Tcatch.isValid();

        if (except)
            tree.addNode(new DExcept(Ttry.getNodes(), Tcatch.getNodes()));
        else {
            // only one of these will add nodes
            tree.addNodes(Ttry.getNodes());
            tree.addNodes(Tcatch.getNodes());
            tree.addNodes(Tfinally.getNodes());
        }


        return tree;
    }
}
