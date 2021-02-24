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

LABELS = ['swing', 'awt', 'security', 'sql', 'net', 'xml', 'crypto', 'math']


def get_api(config, calls, apiOrNot):
    apis = []
    for call, api_bool in zip(calls, apiOrNot):
        if api_bool and call > 0:
            api = config.vocab.chars_api[call]
            apis.append(api)

    apis_ = []
    for api in apis:
        try:
            api_mid = api.split('.')[1]
        except:
            api_mid = []
        apis_.append(api_mid)

    guard = []
    for api in apis_:
        if api in LABELS:
            label = api
            guard.append(label)

    if len(set(guard)) != 1:
        return 'N/A'
    else:
        return guard[0]
