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

class TooLongLoopingException(Exception):
    pass


class TooLongBranchingException(Exception):
    pass


class TooManyTryException(Exception):
    pass

class TooDeepException(Exception):
    pass


class VoidProgramException(Exception):
    pass


class TooManyVariableException(Exception):
    pass


class UnknownVarAccessException(Exception):
    pass


class UndeclaredVarException(Exception):
    pass


class NestedAPIParsingException(Exception):
    pass


class TypeMismatchException(Exception):
    pass


class RetStmtNotExistException(Exception):
    pass


class ConceptMismatchException(Exception):
    pass


class IgnoredForNowException(Exception):
    pass


class UnusedVariableException(Exception):
    pass


class NullPtrException(Exception):
    pass
