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

import java_compiler.dsl.*;
import org.eclipse.jdt.core.dom.VariableDeclarationFragment;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.Name;
import org.eclipse.jdt.core.dom.IVariableBinding;
import org.eclipse.jdt.core.dom.ITypeBinding;


public class DOMVariableDeclarationFragment implements Handler {

    final VariableDeclarationFragment fragment;
    final Visitor visitor;

    public DOMVariableDeclarationFragment(VariableDeclarationFragment fragment, Visitor visitor) {
        this.fragment = fragment;
        this.visitor = visitor;
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();
 

        IVariableBinding ivb = fragment.resolveBinding();
        if ( ivb != null && !ivb.isField() )
            tree.addNode(new DVarCall(ivb));
        else if ( ivb != null && ivb.isField() )
            tree.addNode(new DFieldCall(ivb));

//         if (ivb != null){
//             ITypeBinding temp = ivb.getType();
//             while(temp != temp.getTypeDeclaration()){
//                  temp = temp.getTypeDeclaration();
//             }
//         }


        if (fragment.getInitializer() == null){
            }
        else if (fragment.getInitializer() instanceof Name){
             IVariableBinding _from = fragment.resolveBinding();
             IVariableBinding _to = (IVariableBinding) ((Name) fragment.getInitializer()).resolveBinding();
             
             if (_from != null && _to != null){
                 DAssign Tassign = new DAssign(new DVarCall(_from), new DVarCall(_to));
                 tree.addNode(Tassign);
             }
        }
        else{
            DSubTree Tinit = new DOMExpression(fragment.getInitializer(), visitor).handle();
            tree.addNodes(Tinit.getNodes());
        }

        return tree;
    }
}
