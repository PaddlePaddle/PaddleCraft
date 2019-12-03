# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BaseModel in PaddleCraft."""
import os
import sys
import paddle
import paddle.fluid as fluid


def logical2physical_name(model_name, logical_name):
    return '_'.join([model_name, logical_name])


def physical2logical_name(model_name, physical_name):
    logical_name = physical_name[len(model_name + '_'):]
    return logical_name


class BaseModel(object):
    def __init__(self, name):
        self.name = name

    def get_param(self, param_name, main_program=None):

        raise NotImplementedError(
            "the get_param() method is not implemented yet.")

    def set_param(self, param, value, main_program=None):

        raise NotImplementedError(
            "the set_param() method is not implemented yet.")

    def get_param_gradient(self, var, main_program=None):

        raise NotImplementedError(
            "the get_gradient() method is not implemented yet.")

    def set_param_gradient(self, var, main_program=None):

        raise NotImplementedError(
            "the get_param_gradient() method is not implemented yet.")

    def get_var_gradient(self, var, main_program=None):

        raise NotImplementedError(
            "the get_var_gradient() method is not implemented yet.")

    def build(self):

        raise NotImplementedError("the build() method is not implemented yet.")

    def save(self):

        raise NotImplementedError("the save() method is not implemented yet.")

    def load(self):

        raise NotImplementedError("the load() method is not implemented yet.")

    def print(self):

        raise NotImplementedError("the print() method is not implemented yet.")


if __name__ == "__main__":

    base_model = BaseModel("test_model")

    base_model.get_param("test_param")
