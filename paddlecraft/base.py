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

    def get_param(
        self,
        param_name,
        main_program = None):

        raise NotImplementedError("the get_param() method is not implemented yet.")

    def set_param(
        self,
        param,
        value,
        main_program = None):

        raise NotImplementedError("the set_param() method is not implemented yet.")

    def get_gradient(
        self,
        var,
        main_program = None):
    
        raise NotImplementedError("the get_gradient() method is not implemented yet.")

    def build(self):
        
        raise NotImplementedError("the build() method is not implemented yet.")

    def save(self):
        
        raise NotImplementedError("the save() method is not implemented yet.")

    def load(self):
        
        raise NotImplementedError("the load() method is not implemented yet.")


if __name__ == "__main__":

    base_model = BaseModel("test_model")

    base_model.get_param("test_param")


