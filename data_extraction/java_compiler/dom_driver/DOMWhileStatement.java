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
import org.eclipse.jdt.core.dom.WhileStatement;

public class DOMWhileStatement implements Handler {

    final WhileStatement statement;
    final Visitor visitor;

    public DOMWhileStatement(WhileStatement statement, Visitor visitor) {
        this.statement = statement;
        this.visitor = visitor;
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();

        DSubTree cond = new DOMCondition(statement.getExpression(), visitor).handle();
        DSubTree body = new DOMStatement(statement.getBody(), visitor).handle();

        if (cond.isValid())
            tree.addNode(new DLoop(cond.getNodes(), body.getNodes()));
        else
            tree.addNodes(body.getNodes());

        // Do nothing if both "cond" and "body" are empty

        return tree;
    }
}

