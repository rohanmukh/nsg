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

import org.eclipse.jdt.core.dom.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class DBranch extends DASTNode {

    String node = "DBranch";
    List<DASTNode> _cond;
    List<DASTNode> _then;
    List<DASTNode> _else;

    public DBranch() {
        this._cond = new ArrayList<>();
        this._then = new ArrayList<>();
        this._else = new ArrayList<>();
        this.node = "DBranch";
    }

    public DBranch(List<DASTNode> _cond, List<DASTNode> _then, List<DASTNode> _else) {
        this._cond = _cond;
        this._then = _then;
        this._else = _else;
        this.node = "DBranch";
    }

    @Override
    public void updateSequences(List<Sequence> soFar, int max, int max_length) throws TooManySequencesException, TooLongSequenceException {
        if (soFar.size() >= max)
            throw new TooManySequencesException();
        for (DASTNode call : _cond)
            call.updateSequences(soFar, max, max_length);
        List<Sequence> copy = new ArrayList<>();
        for (Sequence seq : soFar)
            copy.add(new Sequence(seq.calls));
        for (DASTNode t : _then)
            t.updateSequences(soFar, max, max_length);
        for (DASTNode e : _else)
            e.updateSequences(copy, max, max_length);
        for (Sequence seq : copy)
            if (! soFar.contains(seq))
                soFar.add(seq);
    }

    @Override
    public int numStatements() {
        int num = _cond.size();
        for (DASTNode t : _then)
            num += t.numStatements();
        for (DASTNode e : _else)
            num += e.numStatements();
        return num;
    }

    @Override
    public int numLoops() {
        int num = 0;
        for (DASTNode t : _then)
            num += t.numLoops();
        for (DASTNode e : _else)
            num += e.numLoops();
        return num;
    }

    @Override
    public int numBranches() {
        int num = 1; // this branch
        for (DASTNode t : _then)
            num += t.numBranches();
        for (DASTNode e : _else)
            num += e.numBranches();
        return num;
    }

    @Override
    public int numExcepts() {
        int num = 0;
        for (DASTNode t : _then)
            num += t.numExcepts();
        for (DASTNode e : _else)
            num += e.numExcepts();
        return num;
    }


    @Override
    public Set<Class> exceptionsThrown() {
        Set<Class> ex = new HashSet<>();
        for (DASTNode c : _cond)
            ex.addAll(c.exceptionsThrown());
        for (DASTNode t : _then)
            ex.addAll(t.exceptionsThrown());
        for (DASTNode e : _else)
            ex.addAll(e.exceptionsThrown());
        return ex;
    }

    @Override
    public Set<Class> exceptionsThrown(Set<String> eliminatedVars) {
	return this.exceptionsThrown();
    }
    
    @Override
    public boolean equals(Object o) {
        if (o == null || ! (o instanceof DBranch))
            return false;
        DBranch branch = (DBranch) o;
        return _cond.equals(branch._cond) && _then.equals(branch._then) && _else.equals(branch._else);
    }

    @Override
    public int hashCode() {
        return 7* _cond.hashCode() + 17* _then.hashCode() + 31* _else.hashCode();
    }

    @Override
    public String toString() {
        return "if (\n" + _cond + "\n) then {\n" + _then + "\n} else {\n" + _else + "\n}";
    }


}
