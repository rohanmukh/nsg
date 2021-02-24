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

public class DVarCall extends DASTNode
{

    //TODO: public class InvalidVARCallException extends Exception {}

    String node = "DVarCall";
    String _call;
    String _id;
    //List<String> _throws;
    String _returns;
    transient String retVarName = "";

    /* CAUTION: This field is only available during AST generation */
    transient IVariableBinding variableBinding;
    /* CAUTION: These fields are only available during synthesis (after synthesize(...) is called) */
    transient Method method;
    transient Constructor constructor;

    /* TODO: Add refinement types (predicates) here */

    public DVarCall() {
        //this._call = "";
        this.node = "DVarCall";
        this._id = "null";
        this._returns = null;
    }

    public DVarCall(String id) {
        //this._call = "";
        this.node = "DVarCall";
        this._id = id;
        this._returns = null;
    }

    public DVarCall(IVariableBinding variableBinding) {
        this.variableBinding = variableBinding;
        String prefix;
        if (variableBinding != null){
            if (variableBinding.isField())
               prefix = "field_";
            else
               prefix = "local_";
            this._id = prefix + String.valueOf(variableBinding.getVariableId()) ; 
            this._returns = getClassName(variableBinding);
        }
        this.node = "DVarCall";
    }

    public String get_id_value(){
        return this._id;
    }

    public String get_type_value(){
        return this._returns;
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

    public static String getClassName(IVariableBinding variableBinding) {
        ITypeBinding temp_1 = variableBinding.getType();
        ITypeBinding temp_2 = temp_1.getTypeDeclaration();
        String className = temp_2.getQualifiedName();
        if (temp_1.isParameterizedType())
            className += "<" + String.join(",", Arrays.stream(temp_2.getTypeParameters()).map(
                    t -> getTypeName(t, t.getName())
            ).collect(Collectors.toList())) + ">";
        else if (temp_1.isFromSource() || temp_1.isLocal() || temp_1.isMember() || temp_1.isNested())
            className = "Tau_K";
        return className;
    }

    //private String getSignature() throws InvalidAPICallException {
    //    Stream<String> types = Arrays.stream(methodBinding.getParameterTypes()).map(
    //            t -> getTypeName(t, t.getQualifiedName()));
    //    if (methodBinding.getName().equals(""))
    //        throw new InvalidAPICallException();
    //    return methodBinding.getName() + "(" + String.join(",", types.collect(Collectors.toCollection(ArrayList::new))) + ")";
    //}

    public static String getTypeName(ITypeBinding binding, String name) {
        //ITypeBinding temp = binding.getTypeDeclaration();
        //return temp.getQualifiedName();
        return (binding.isTypeVariable()? "Tau_" : "") + name;
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
        if (o == null || ! (o instanceof DAPICall))
            return false;
        DAPICall apiCall = (DAPICall) o;
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

}
