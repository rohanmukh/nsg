# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from program_helper.ast.ops.node import Node, CHILD_EDGE, SIBLING_EDGE

from program_helper.ast.ops.concepts.DStop import DStop
from program_helper.ast.ops.concepts.DSubTree import DSubTree

from program_helper.ast.ops.control_flow.DBranch import DBranch, DCond, DThen, DElse
from program_helper.ast.ops.control_flow.DExcept import DExcept, DTry, DCatch
from program_helper.ast.ops.control_flow.DLoop import DLoop, DBody

from program_helper.ast.ops.leaf_ops.DAPICall import DAPICall
from program_helper.ast.ops.leaf_ops.DVarAccess import DVarAccess
from program_helper.ast.ops.leaf_ops.DVarAccessDecl import DVarAccessDecl
from program_helper.ast.ops.leaf_ops.DType import DType
from program_helper.ast.ops.leaf_ops.DSymtabMod import DSymtabMod

from program_helper.ast.ops.concepts.DVarDecl import DVarDecl
from program_helper.ast.ops.concepts.DAPIInvoke import DAPIInvoke
from program_helper.ast.ops.concepts.DFieldDecl import DFieldDecl
from program_helper.ast.ops.concepts.DClsInit import DClsInit
from program_helper.ast.ops.concepts.DVarAssign import DVarAssign
from program_helper.ast.ops.concepts.DAPICallMulti import DAPICallMulti
from program_helper.ast.ops.concepts.DAPICallSingle import DAPICallSingle
from program_helper.ast.ops.concepts.DInfix import DInfix, DLeft, DRight
from program_helper.ast.ops.concepts.DReturnVar import DReturnVar
from program_helper.ast.ops.concepts.DVarDeclCls import DVarDeclCls

# CONCEPT_NODE_NAMES = [DAPIInvoke.name(), DClsInit.name(), DVarAssign.name(), DVarDecl.name()]
CONTROL_FLOW_NAMES = [DBranch.name(), DLoop.name(), DExcept.name()]

SINGLE_PARENTS = [DSubTree.name(), DCond.name(),
                  DBody.name(), DThen.name(), DElse.name(),
                  DTry.name(), DCatch.name(),
                  DLeft.name(), DRight.name(),
                  DAPICallMulti.name()
                  ]

MAJOR_CONCEPTS = [DAPIInvoke.name(), DClsInit.name(), DFieldDecl.name(), DVarDeclCls.name(),
                  DVarDecl.name(), DVarAssign.name(), DAPICallSingle.name(), DInfix.name(),
                  DReturnVar.name(), DStop.name()
                  ]
