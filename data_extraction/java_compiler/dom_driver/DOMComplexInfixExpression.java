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

import java.util.*;
import java_compiler.dsl.DSubTree;
import java_compiler.dsl.DInfix;
import org.eclipse.jdt.core.dom.InfixExpression;


public class DOMComplexInfixExpression implements Handler {

    final InfixExpression expr;
    final Visitor visitor;
    final List<String> listOfOps;

    public DOMComplexInfixExpression(InfixExpression expr, Visitor visitor) {
        this.expr = expr;
        this.visitor = visitor;
        this.listOfOps = new ArrayList<String>( 
            Arrays.asList("<", ">", "<=", ">=", "==", "!=", "&&", "||", "^", "|", "&")); 
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();

        DSubTree Tleft = new DOMCondition(expr.getLeftOperand(), visitor).handle();
        DSubTree Tright = new DOMCondition(expr.getRightOperand(), visitor).handle();
        String op = expr.getOperator().toString();

        if ((Tleft.isValid() || Tright.isValid()) &&  listOfOps.contains(op) )
             tree.addNode(new DInfix(Tleft.getNodes(), Tright.getNodes(), op));
        else if (Tleft.isValid())
             tree.addNodes(Tleft.getNodes());
        else if (Tright.isValid())
             tree.addNodes(Tright.getNodes());

        return tree;
    }
}
