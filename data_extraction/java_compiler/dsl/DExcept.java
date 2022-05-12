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

import java.util.*;

public class DExcept extends DASTNode {

    String node = "DExcept";
    List<DASTNode> _try;
    List<DASTNode> _catch;
    transient Map exceptToClause;

    public DExcept() {
        this._try = new ArrayList<>();
        this._catch = new ArrayList<>();
        this.node = "DExcept";
    }

    public DExcept(List<DASTNode> _try, List<DASTNode> _catch) {
        this._try = _try;
        this._catch = _catch;
        this.node = "DExcept";
    }

    @Override
    public void updateSequences(List<Sequence> soFar, int max, int max_length)  throws TooManySequencesException, TooLongSequenceException {
        if (soFar.size() >= max)
            throw new TooManySequencesException();
        for (DASTNode node : _try)
            node.updateSequences(soFar, max, max_length);
        List<Sequence> copy = new ArrayList<>();
        for (Sequence seq : soFar)
            copy.add(new Sequence(seq.calls));
        for (DASTNode e : _catch)
            e.updateSequences(copy, max, max_length);
        for (Sequence seq : copy)
            if (! soFar.contains(seq))
                soFar.add(seq);
    }

    @Override
    public int numStatements() {
        int num = _try.size();
        for (DASTNode c : _catch)
            num += c.numStatements();
        return num;
    }

    @Override
    public int numLoops() {
        int num = 0;
        for (DASTNode t : _try)
            num += t.numLoops();
        for (DASTNode c : _catch)
            num += c.numLoops();
        return num;
    }

    @Override
    public int numBranches() {
        int num = 0;
        for (DASTNode t : _try)
            num += t.numBranches();
        for (DASTNode c : _catch)
            num += c.numBranches();
        return num;
    }

    @Override
    public int numExcepts() {
        int num = 1; // this except
        for (DASTNode t : _try)
            num += t.numExcepts();
        for (DASTNode c : _catch)
            num += c.numExcepts();
        return num;
    }


    @Override
    public Set<Class> exceptionsThrown() {
        Set<Class> ex = new HashSet<>();
        // no try: whatever thrown in try would have been caught in catch
        for (DASTNode c : _catch)
            ex.addAll(c.exceptionsThrown());
        return ex;
    }

    @Override
    public Set<Class> exceptionsThrown(Set<String> eliminatedVars) {
        return this.exceptionsThrown();
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || ! (o instanceof DExcept))
            return false;
        DExcept other = (DExcept) o;
        return _try.equals(other._try) && _catch.equals(other._catch);
    }

    @Override
    public int hashCode() {
        return 7* _try.hashCode() + 17* _catch.hashCode();
    }

    @Override
    public String toString() {
        return "try {\n" + _try + "\n} catch {\n" + _catch + "\n}";
    }

    public void cleanupCatchClauses(Set<String> eliminatedVars) {
        Set excepts = new HashSet<>();
        for (DASTNode tn : _try) {
            if (tn instanceof DAPICall) {
                String retVarName = ((DAPICall)tn).getRetVarName();
                if (!retVarName.equals("") && eliminatedVars.contains(retVarName))
                    excepts.addAll(tn.exceptionsThrown());
            }
        }

        for (Object obj : this.exceptToClause.keySet()) {
            if (excepts.contains(obj)) {
                CatchClause catchClause = (CatchClause)this.exceptToClause.get(obj);
                catchClause.delete();
            }
        }
    }

}
