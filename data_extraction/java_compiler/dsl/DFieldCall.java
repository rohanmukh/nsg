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

public class DFieldCall extends DVarCall
{


    public DFieldCall() {
        //this._call = "";
        this.node = "DFieldCall";
    }

    public DFieldCall(IVariableBinding variableBinding) {
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
        this.node = "DFieldCall";
    }


}
