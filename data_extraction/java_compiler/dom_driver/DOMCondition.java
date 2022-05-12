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
import org.eclipse.jdt.core.dom.*;


public class DOMCondition implements Handler {

    final Expression expression;
    final Visitor visitor;

    public DOMCondition(Expression expression, Visitor visitor) {
        this.expression = expression;
        this.visitor = visitor;
    }

    @Override
    public DSubTree handle() {

        if (expression instanceof MethodInvocation)
            return new DOMMethodInvocation((MethodInvocation) expression, visitor).handle();
        if (expression instanceof InfixExpression)
            return new DOMComplexInfixExpression((InfixExpression) expression, visitor).handle();
        if (expression instanceof ParenthesizedExpression)
            return new DOMCondition(((ParenthesizedExpression) expression).getExpression(), visitor).handle();
        return new DSubTree();
    }
}
