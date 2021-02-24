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

public class DClsInit extends DASTNode {
    public class InvalidAPICallException extends Exception {}

    String node = "DClsInit";
    String _call;
    List<DASTNode> fp_var_ids;
    String _var;
    String _type;
    String _returns;

    public DClsInit() {
        this._call = null;
        this.fp_var_ids = null;
        this._var = null;
        this._returns = null;
        this.node = "DClsInit";
    }

    public DClsInit(DAPICall call, DVarCall var){
        this._call = call.get_call_value();
        this.fp_var_ids = call.get_arglist_value();
        this._var = var.get_id_value();
        this._type = call.getClassName();
        this._returns = this._type; //var.get_type_value(); //call.get_ret_value();
                      
        this.node = "DClsInit";
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
        if (o == null || ! (o instanceof DBranch))
            return false;
        DClsInit cls = (DClsInit) o;
        return _call.equals(cls._call) && _var.equals(cls._var); 
    }

    @Override
    public int hashCode() {
        return 7* _call.hashCode() + 7* _var.hashCode();
    }

    @Override
    public String toString() {
        return "Not done";
    }


}
