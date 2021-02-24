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

public class DVarDeclExpr extends DASTNode {

    String node = "DVarDeclExpr";
    List<DASTNode> _stmt;
    String id = null;

    public DVarDeclExpr() {
        this._stmt = new ArrayList<>();
        this.node = "DVarDeclExpr";
    }

    public DVarDeclExpr(List<DASTNode> _stmt) {
        this._stmt = _stmt;
        this.node = "DVarDeclExpr";
    }

    public DVarDeclExpr(DASTNode node) {
        this._stmt = new ArrayList<DASTNode>();
        this._stmt.add(node);
        this.node = "DVarDeclExpr";
    }

    @Override
    public void updateSequences(List<Sequence> soFar, int max, int max_length) throws TooManySequencesException, TooLongSequenceException {
        if (soFar.size() >= max)
            throw new TooManySequencesException();
    }

    @Override
    public int numStatements() {
        return 0;
    }

    @Override
    public int numLoops() {
        int num = 0;
        return num;
    }

    @Override
    public int numBranches() {
        int num = 1; // this branch
        return num;
    }

    @Override
    public int numExcepts() {
        int num = 0;
        return num;
    }


    @Override
    public Set<Class> exceptionsThrown() {
        Set<Class> ex = new HashSet<>();
        return ex;
    }

    @Override
    public Set<Class> exceptionsThrown(Set<String> eliminatedVars) {
	return this.exceptionsThrown();
    }
    
    @Override
    public boolean equals(Object o) {
        //if (o == null || ! (o instanceof DBranch))
        return false;
        //DBranch branch = (DBranch) o;
        //return _cond.equals(branch._cond) && _then.equals(branch._then) && _else.equals(branch._else);
    }

    @Override
    public int hashCode() {
        return 7; //* _cond.hashCode() + 17* _then.hashCode() + 31* _else.hashCode();
    }

    @Override
    public String toString() {
        return "ret"; // (\n" + _cond + "\n) then {\n" + _then + "\n} else {\n" + _else + "\n}";
    }


}
