import os
import sys
import numpy as np
import paddle
import paddle.fluid as fluid

from paddle.fluid.framework import Parameter, Variable

from paddlecraft.base import logical2physical_name, physical2logical_name
from paddlecraft.base import BaseModel

class ImageMLPEncoder(BaseModel):

    def __init__(self, name, 
        image_shape = [1, 28, 28], 
        num_classes = 10):

        self.name = name
        self.num_classes = num_classes
        self.model = {
                      'image':logical2physical_name(self.name, 'image'),
                      'label':logical2physical_name(self.name, 'label'),
                      'hidden_layer_1':logical2physical_name(self.name, 'hidden1'),
                      'hidden_layer_2':logical2physical_name(self.name, 'hidden2'),
                      'prediction':logical2physical_name(self.name, 'prediction'),
                     }

    def get_param(self, 
        param_name,
        main_program = None):
        
        _main_program = main_program if main_program is not None else fluid.default_main_program()
        
        # the match params will store the parameters whose name has the prefix of param_name
        matched_params = {}
        physical_param_name = self.model[param_name]

        # the parameter is stored in the global block by default
        # fix this if we are going to change something in the future
        for _var_physical_name in _main_program.global_block().vars:
            if not isinstance(_main_program.global_block().vars[_var_physical_name], Parameter):
                continue
            if _var_physical_name.startswith(physical_param_name):
                matched_param = _main_program.global_block().vars[_var_physical_name]
                if param_name not in matched_params:
                    matched_params[param_name] = []
                matched_params[param_name].append(matched_param)

        return matched_params

    def get_gradient(self,
        var,
        main_program = None):

        assert (isinstance(var, Variable) or isinstance(var, type('')) or isinstance(var, type(u'')))

        if not isinstance(var, Variable):
            if var in self.model:
                var_physical_name = self.model[var]
            else:
                var_physical_name = logical2physical_name(self.name, var)
        else:
            var_physical_name = var.name
       
        _main_program = main_program if main_program is not None else fluid.default_main_program()

        matched_grads = {}

        for _block in _main_program.blocks:
            for _var_name in _block.vars:
                if _var_name.startswith(var_physical_name) and _var_name.endswith('@GRAD'):
                    var_logical_name = physical2logical_name(self.name, _var_name)
                    if var_logical_name not in matched_grads:
                        matched_grads[var_logical_name] = []
                    matched_grads[var_logical_name].append(_block.vars[_var_name])

        return matched_grads

    def build(self):

        self.input = fluid.data(name=self.model['image'], shape=[-1, 784], dtype='float32')
        self.label = fluid.data(name=self.model['label'], shape=[-1, 1], dtype='int64')

        self.image1 = fluid.layers.fc(
            input=self.input, size=784, act='relu', num_flatten_dims = 1, 
            param_attr=fluid.ParamAttr(name = '.'.join([self.model['hidden_layer_1'], 'w_0'])),
            bias_attr='.'.join([self.model['hidden_layer_1'], 'b_0']))

        self.image2 = fluid.layers.fc(
            input=self.image1, size=784, act='relu', num_flatten_dims = 1,
            param_attr=fluid.ParamAttr(name = '.'.join([self.model['hidden_layer_2'], 'w_0'])),
            bias_attr='.'.join([self.model['hidden_layer_2'], 'b_0']))

        self.prediction = fluid.layers.fc(input=self.image2, size=self.num_classes, act='softmax',
            param_attr=fluid.ParamAttr(name = '.'.join([self.model['prediction'], 'w_0'])),
            bias_attr='.'.join([self.model['prediction'], 'b_0']))


    def save(self):
        pass

    def load(self):
        pass


if __name__ == "__main__":
    # start model configuration
    image_encoder1 = ImageMLPEncoder('model1')
    print(image_encoder1.model)

    image_encoder1.build()

    # test build loss and optimizer

    loss = fluid.layers.cross_entropy(input=image_encoder1.prediction, label=image_encoder1.label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.RMSProp(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    # test fetching parameters
    print("="*50 + "show parameters" + "="*50)
    params = image_encoder1.get_param('hidden_layer_1')

    for _var_name in params:
        for param in params[_var_name]:
            print(_var_name, param.name)

    # test fetching gradient
    print("="*50 + "show gradients" + "="*50)
    grads = image_encoder1.get_gradient('hidden_layer_1')
    for gard_var_name in grads:
        for grad_vars in grads[gard_var_name]:
            print(gard_var_name, grad_vars.name)

    # test freezing/unfreezing

    image_encoder2 = ImageMLPEncoder('model2')
    # test sharing parameters
    image_encoder2.model['hidden_layer_1'] = image_encoder1.model['hidden_layer_1']

