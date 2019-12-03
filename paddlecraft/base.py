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


def convert2gradname(name):
    return name + "@GRAD"


def logical2physical_name(model_name, logical_name):
    return '_'.join([model_name, logical_name])


def physical2logical_name(model_name, physical_name):
    logical_name = physical_name[len(model_name + '_'):]
    return logical_name


class BaseModel(object):
    def __init__(self, name):
        self.name = name

    def get_param(self, param_name, main_program=None):

        assert (isinstance(param_name, type('')) or isinstance(param_name,
                                                               type(u'')))

        _main_program = main_program if main_program is not None else fluid.default_main_program(
        )

        if param_name not in _main_program.global_block().vars:
            raise Warning("[WARNING] the param %s is not in the program" %
                          param_name)
            return None

        return _main_program.global_block().vars[param_name]

    def set_param(self, param, value, main_program=None):

        raise NotImplementedError(
            "the set_param() method is not implemented yet.")

    def get_param_gradient(self, param_name, main_program=None):

        assert (isinstance(param_name, type('')) or isinstance(param_name,
                                                               type(u'')))

        _main_program = main_program if main_program is not None else fluid.default_main_program(
        )

        param_grad_name = convert2gradname(param_name)

        for _block in _main_program.blocks:
            if param_grad_name in _block.vars:
                return _block.vars[param_grad_name]

        return None

    def set_param_gradient(self, param_name, value, main_program=None):

        raise NotImplementedError(
            "the set_param_gradient() method is not implemented yet.")

    def get_var_gradient(self, var, main_program=None):

        assert isinstance(var, Variable)

        _main_program = main_program if main_program is not None else fluid.default_main_program(
        )

        var_name = var.name

        return self.get_param_gradient(var_name, main_program=_main_program)

    def set_var_gradient(self, var, value, main_program=None):

        raise NotImplementedError(
            "the set_var_gradient() method is not implemented yet.")

    def build(self):

        raise NotImplementedError("the build() method is not implemented yet.")

    def save(self):

        raise NotImplementedError("the save() method is not implemented yet.")

    def load(self):

        raise NotImplementedError("the load() method is not implemented yet.")

    def print_model(self):

        raise NotImplementedError("the print() method is not implemented yet.")


if __name__ == "__main__":

    base_model = BaseModel("test_model")
    base_model.get_param("test_param")
