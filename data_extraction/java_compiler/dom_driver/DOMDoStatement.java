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

import java_compiler.dsl.DLoop;
import java_compiler.dsl.DSubTree;
import org.eclipse.jdt.core.dom.DoStatement;

public class DOMDoStatement implements Handler {

    final DoStatement statement;
    final Visitor visitor;

    public DOMDoStatement(DoStatement statement, Visitor visitor) {
        this.statement = statement;
        this.visitor = visitor;
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();

        DSubTree cond = new DOMExpression(statement.getExpression(), visitor).handle();
        DSubTree body = new DOMStatement(statement.getBody(), visitor).handle();

        boolean loop = cond.isValid() && body.isValid();

        tree.addNodes(body.getNodes());
        if (loop)
            tree.addNode(new DLoop(cond.getNodes(), body.getNodes()));
        else {
            // only one of these will add nodes
            tree.addNodes(cond.getNodes());
            tree.addNodes(body.getNodes());
        }

        return tree;
    }
}
