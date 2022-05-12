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

import java.lang.reflect.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DInternalAPICall extends DASTNode
{

    public class InvalidAPICallException extends Exception {}

    String node = "DInternalAPICall";
    String _call;
    List<String> _throws;
    String _returns;
    transient String retVarName = "";

    List<DASTNode> fp_var_ids;
    String ret_var_id;
    //DASTNode expr_var_id;

    /* CAUTION: This field is only available during AST generation */
    String int_meth_name;
    /* CAUTION: These fields are only available during synthesis (after synthesize(...) is called) */
    transient Method method;
    transient Constructor constructor;

    /* TODO: Add refinement types (predicates) here */

    public DInternalAPICall() {
        this._call = null;
        this.node = "DInternalAPICall";
    }


    public DInternalAPICall(String int_meth_name, List<DASTNode> fp_var_ids, DVarCall ret_var) {
        this.int_meth_name = int_meth_name;
        this.fp_var_ids = new ArrayList<DASTNode>();
        for (DASTNode i : fp_var_ids)
            this.fp_var_ids.add(i);
        this.ret_var_id = ret_var.get_id_value();
        this.node = "DInternalAPICall";
    }

    public String get_ret_value(){
        return this._returns;
    }

    public String get_call_value(){
        return this._call;
    }

    public List<DASTNode> get_arglist_value(){
        return this.fp_var_ids;
    }

    @Override
    public void updateSequences(List<Sequence> soFar, int max, int max_length) throws TooManySequencesException, TooLongSequenceException {
        if (soFar.size() >= max)
            throw new TooManySequencesException();
        for (Sequence sequence : soFar) {
            sequence.addCall(_call);
            if (sequence.getCalls().size() > max_length)
                throw new TooLongSequenceException();
        }
    }


    public void setNotPredicate() {
        this._call = "$NOT$" + this._call;
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
        if (constructor != null)
            return new HashSet<>(Arrays.asList(constructor.getExceptionTypes()));
        else
            return new HashSet<>(Arrays.asList(method.getExceptionTypes()));
    }

    @Override
    public Set<Class> exceptionsThrown(Set<String> eliminatedVars) {
        if (!eliminatedVars.contains(this.retVarName))
            return this.exceptionsThrown();
        else
            return new HashSet<>();
    }

    public String getRetVarName() {
        return this.retVarName;
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || ! (o instanceof DInternalAPICall))
            return false;
        DInternalAPICall apiCall = (DInternalAPICall) o;
        return _call.equals(apiCall._call);
    }

    @Override
    public int hashCode() {
        return _call.hashCode();
    }

    @Override
    public String toString() {
        return _call;
    }



    /**
     * Returns the name of a given executable from its toString() method
     *
     * @param e the executable
     * @return the name of the executable
     */
    private String getNameAsString(Executable e) {
        for (String s : e.toString().split(" "))
            if (s.contains("("))
                return s.replaceAll("\\$", ".");
        return null;
    }


    private boolean hasTypeVariable(String className) {
        if (className.contains("Tau_"))
            return true;

        // commonly used type variable names in Java API
        Matcher typeVars = Pattern.compile("\\b[EKNTVSU][0-9]?\\b").matcher(className);
        return typeVars.find();
    }
}
