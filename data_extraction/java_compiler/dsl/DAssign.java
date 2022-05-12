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

public class DAssign extends DASTNode {

    String node = "DAssign";
    String _from;
    String _to;
    String _type;

    public DAssign() {
        this._from = null;
        this._to = null;
        this._type = null;
        this.node = "DAssign";
    }

    public DAssign(DVarCall _from, DVarCall _to) {
        this._from = _from.get_id_value();
        this._to = _to.get_id_value();
        this._type = _from.get_type_value();
        this.node = "DAssign";
    }

    @Override
    public void updateSequences(List<Sequence> soFar, int max, int max_length) throws TooManySequencesException, TooLongSequenceException {
        if (soFar.size() >= max)
            throw new TooManySequencesException();
    }


    @Override
    public int numStatements() {
        return 1;
    }

    @Override
    public int numLoops() {
        return 0;
    }

    @Override
    public int numBranches() {
        return 0;
    }

    @Override
    public int numExcepts() {
        return 0;
    }

    @Override
    public Set<Class> exceptionsThrown() {
        return new HashSet<>();
    }

    @Override
    public Set<Class> exceptionsThrown(Set<String> eliminatedVars) {
        return new HashSet<>();
    }

    public String getRetVarName() {
        return "";
    }

    
    @Override
    public boolean equals(Object o) {
        return false; //_from.equals(cls._from) && _v.equals(cls._var); 
    }

    @Override
    public int hashCode() {
        return 7* _from.hashCode() + 7* _to.hashCode();
    }

    @Override
    public String toString() {
        return "Not done";
    }


}
