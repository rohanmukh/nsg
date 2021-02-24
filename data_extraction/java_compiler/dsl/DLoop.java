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
package java_compiler.dsl;

import java_compiler.dom_driver.Visitor;
import org.eclipse.jdt.core.dom.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class DLoop extends DASTNode {

    String node = "DLoop";
    List<DASTNode> _cond;
    List<DASTNode> _body;

    public DLoop() {
        this._cond = new ArrayList<>();
        this._body = new ArrayList<>();
        this.node = "DLoop";
    }

    public DLoop(List<DASTNode> cond, List<DASTNode> _body) {
        this._cond = cond;
        this._body = _body;
        this.node = "DLoop";
    }

    @Override
    public void updateSequences(List<Sequence> soFar, int max, int max_length) throws TooManySequencesException, TooLongSequenceException {
        if (soFar.size() >= max)
            throw new TooManySequencesException();
        for (DASTNode call : _cond)
            call.updateSequences(soFar, max, max_length);

        int num_unrolls = 1;
        for (int i = 0; i < num_unrolls; i++) {
            for (DASTNode node : _body)
                node.updateSequences(soFar, max, max_length);
            for (DASTNode call : _cond)
                call.updateSequences(soFar, max, max_length);
        }
    }

    @Override
    public int numStatements() {
        int num = _cond.size();
        for (DASTNode b : _body)
            num += b.numStatements();
        return num;
    }

    @Override
    public int numLoops() {
        int num = 1; // this loop
        for (DASTNode b : _body)
            num += b.numLoops();
        return num;
    }

    @Override
    public int numBranches() {
        int num = 0;
        for (DASTNode b : _body)
            num += b.numBranches();
        return num;
    }

    @Override
    public int numExcepts() {
        int num = 0;
        for (DASTNode b : _body)
            num += b.numExcepts();
        return num;
    }


    @Override
    public Set<Class> exceptionsThrown() {
        Set<Class> ex = new HashSet<>();
        for (DASTNode c : _cond)
            ex.addAll(c.exceptionsThrown());
        for (DASTNode b : _body)
            ex.addAll(b.exceptionsThrown());
        return ex;
    }

    @Override
    public Set<Class> exceptionsThrown(Set<String> eliminatedVars) {
	return this.exceptionsThrown();
    }	

    @Override
    public boolean equals(Object o) {
        if (o == null || ! (o instanceof DLoop))
            return false;
        DLoop loop = (DLoop) o;
        return _cond.equals(loop._cond) && _body.equals(loop._body);
    }

    @Override
    public int hashCode() {
        return 7* _cond.hashCode() + 17* _body.hashCode();
    }

    @Override
    public String toString() {
        return "while (\n" + _cond + "\n) {\n" + _body + "\n}";
    }


}
