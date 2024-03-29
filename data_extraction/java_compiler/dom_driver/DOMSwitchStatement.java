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

import java_compiler.dsl.DASTNode;
import java_compiler.dsl.DBranch;
import java_compiler.dsl.DSubTree;
import org.eclipse.jdt.core.dom.Statement;
import org.eclipse.jdt.core.dom.SwitchStatement;

public class DOMSwitchStatement implements Handler {

	final SwitchStatement statement;
	final Visitor visitor;

	DSubTree tree = new DSubTree();

	ArrayList<List<DASTNode>> bodies = new ArrayList<>();
	ArrayList<Integer> nodeType = new ArrayList<>();

	public DOMSwitchStatement(SwitchStatement statement, Visitor visitor) {
		this.statement = statement;
		this.visitor = visitor;
	}

	private DSubTree BuildTree(DSubTree Sexpr, int itPos) {
		DSubTree bodyPrev = new DSubTree();
		DSubTree caseNodes = new DSubTree();
                DSubTree bodyNext;
		for (int it1 = itPos; it1 < bodies.size(); it1++) {
			int typePrev = nodeType.get(it1);
			if (typePrev == 49) { // checks for 'case' statement
				bodyNext = BuildTree(Sexpr, it1+1);
                                if (Sexpr.isValid()){
				    DASTNode caseNode = new DBranch(Sexpr.getNodes(), bodyPrev.getNodes(), bodyNext.getNodes());
			            caseNodes.addNode(caseNode);
                                }
				return caseNodes;
			} else {
				bodyPrev.addNodes(bodies.get(it1));
			}
		}

		return bodyPrev;
	}

	@Override
	public DSubTree handle() {
		DSubTree Sexpr = new DOMExpression(statement.getExpression(), visitor).handle();
		boolean branch = Sexpr.isValid();

		for (Object o : statement.statements()) {
			int type = ((Statement) o).getNodeType();
			nodeType.add(type);
			DSubTree body = new DOMStatement((Statement) o, visitor).handle();
			bodies.add(body.getNodes());
			if (type != 49) // excludes 'case' statement
				branch |= body.isValid();
		}

		if (branch) {
			DSubTree switchNode = BuildTree(Sexpr, 1);
			tree.addNode(switchNode.getNodes().get(0));
		} else {
			// only one  will add nodes, the rest will add nothing
			tree.addNodes(Sexpr.getNodes());
			for (Iterator<Object> iter = statement.statements().iterator(); iter.hasNext(); ) {
				Object o = iter.next();
				DSubTree body = new DOMStatement((Statement) o, visitor).handle();
				tree.addNodes(body.getNodes());
			}
		}

		return tree;
	}
}
