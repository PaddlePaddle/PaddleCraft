# PaddleCraft

A lib of models for human-like AI.



### 什么是PaddleCraft

PaddleCraft是一套基于模型的API，使用PaddleCraft，您可以直接声明并加载各种CV，Speech，NLP等经典人工智能领域的模型（及其预训练参数），并可以通过各种预置功能，对不同模型进行链接，对抗等操作。

PaddleCraft完全基于PaddlePaddle平台开发；



### 为什么要提出PaddleCraft

借助于算力，大数据，模型工程的不断提升，经典领域的NLP，CV，Speech模型已经逐渐收敛到一些固定结构和范式。虽然已有可以解决特定领域的问题，然而往往仍需要针对特点问题进行加工打磨才能达到商用的目的；另外，由于模型工程的收敛，解决经典场景任务（如文本分类，图像识别）的门槛越来越低，造成竞争白热化。

为了更好的发展，我们需要能够解决更复杂场景，多模态下的AI问题，因此，我们创造了PaddleCraft。



### PaddleCraft能解决什么问题

PaddleCraft为开发者提供了各领域常见且典型的神经网络模型。不同于大部分模型库，PaddleCraft中的模型都以API的形式存在；我们针对各种行业标准化的模型做了细致的整理和加工，使得开发者可以无需了解模型细节，就可以快速使用，解决80%的常见问题。

除此以外，PaddleCraft统一了各个行业解决标准模型的使用方式和数据流，使得不同领域的模型可以相互协作，对抗，亦或者组成一个新的模型。

使用PaddleCraft可以显著减少开发多模态，多智能体等不同场景的AI代码，加快对解决复杂AI任务的研发周期。



### 如何使用PaddleCraft

下面我们以MNIST图像识别模型（image_mlp_encoder）为例，讲解如何使用PaddleCraft：



假设我们现在的行业标准模型为： 



![Image result for MLP mnist](https://corochann.com/wp-content/uploads/2017/02/mlp-800x367.png)



如上图所示，这是一个MLP模型，假设两个隐层的size都为784；那么我们如何使用这个模型并完成一些操作呢？在PaddleCraft中，每个模型（以及模型中的op，param）是通过2个名字来控制的：

1. 逻辑名字：所谓逻辑名字，就是这个模型在算法描述中的客观名字，比如在上述模型中，我们用 hidden_layer_1, hidden_layer_2，prediction等名字来描述这个模型中的不同layer。
2. 物理名字：所谓物理名字，就是这个模型的op在真是硬件环境中的名字，比如paddle.model1.hidden_layer_1.fc_0.w_0。

过去我们大部分人在使用paddle过程中的主要痛苦，源于需要操作物理名字，而缺乏直接手段找到，并通过物理名字操作op和var，这就导致二次开发成本很高。因此，PaddleCraft在设计的时候，设计了一套直接通过逻辑名字就可以操作模型的API，但是仍然返回，并支持使用物理名字对program进行操作；



我们以下面的例子来说明如何使用PaddleCraft



首先安装PaddleCraft：

```shell
pip install paddlecraft
```





##### case1：常规训练

```python
import os
import sys
import paddle
import paddle.fluid as fluid

import paddlecraft

from paddlecraft.image_mlp_encoder import ImageMLPEncoder


if __name__ == "__main__":
    
    # build dataset
    mnist_dataset_train = paddle.dataset.mnist.train()
    mnist_dataset_test = paddle.dataset.mnist.test()

    # configure the network

    image_encoder1 = ImageMLPEncoder('model1')
    image_encoder1.build()

    loss = fluid.layers.cross_entropy(input=image_encoder1.prediction, label=image_encoder1.label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.RMSProp(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    # build reader
    train_reader = paddle.batch(paddle.reader.shuffle(mnist_dataset_train, buf_size=500), batch_size=128)

    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[image_encoder1.input, image_encoder1.label], place=place)

    # prepare running
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program()) 

    for epoch in range(3):
        for step_id, data in enumerate(train_reader()):
            output = exe.run(fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_loss])
            if step_id % 100 == 0:
                print("training step %d, training loss %.3f" % (step_id, output[0]))


```



case2：共享参数

```python
import os
import sys
import paddle
import paddle.fluid as fluid

from paddlecraft.image_mlp_encoder import ImageMLPEncoder

if __name__ == "__main__":
    
    # build dataset
    mnist_dataset_train = paddle.dataset.mnist.train()
    mnist_dataset_test = paddle.dataset.mnist.test()

    # configure the network

    image_encoder1 = ImageMLPEncoder('model1')
    image_encoder1.model['hidden_layer_2'] = image_encoder1.model['hidden_layer_1']
    
    image_encoder1.build()

    loss = fluid.layers.cross_entropy(input=image_encoder1.prediction, label=image_encoder1.label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.RMSProp(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    # build reader
    train_reader = paddle.batch(paddle.reader.shuffle(mnist_dataset_train, buf_size=500), batch_size=128)

    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[image_encoder1.input, image_encoder1.label], place=place)

    # prepare running
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program()) 

    for epoch in range(3):
        for step_id, data in enumerate(train_reader()):
            output = exe.run(fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_loss])
            if step_id % 100 == 0:
                print("training step %d, training loss %.3f" % (step_id, output[0]))

```



case3：获取参数

```python
import os
import sys
import paddle
import paddle.fluid as fluid

from paddlecraft.image_mlp_encoder import ImageMLPEncoder


if __name__ == "__main__":
    
    # build dataset
    mnist_dataset_train = paddle.dataset.mnist.train()
    mnist_dataset_test = paddle.dataset.mnist.test()

    # configure the network

    image_encoder1 = ImageMLPEncoder('model1')
    image_encoder1.model['hidden_layer_2'] = image_encoder1.model['hidden_layer_1']
    
    image_encoder1.build()

    loss = fluid.layers.cross_entropy(input=image_encoder1.prediction, label=image_encoder1.label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.RMSProp(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    # test fetching parameters
    print("="*50 + "show parameters" + "="*50)
    params = image_encoder1.get_param('hidden_layer_1')

    fetch_params = [avg_loss]
    for _var_name in params:
        for param in params[_var_name]:
            print(_var_name, param.name)
            fetch_params.append(param)

    # build reader
    train_reader = paddle.batch(paddle.reader.shuffle(mnist_dataset_train, buf_size=500), batch_size=128)

    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[image_encoder1.input, image_encoder1.label], place=place)

    # prepare running
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program()) 

    for epoch in range(3):
        for step_id, data in enumerate(train_reader()):
            output = exe.run(fluid.default_main_program(), feed=feeder.feed(data), fetch_list=fetch_params)
            if step_id % 100 == 0:
                print("training step %d, training loss %.3f" % (step_id, output[0]))
            if step_id % 300 == 0:
                for i in range(1, len(output)):
                    print(output[i])


```



case4：获取梯度

```python
import os
import sys
import paddle
import paddle.fluid as fluid

from paddlecraft.image_mlp_encoder import ImageMLPEncoder


if __name__ == "__main__":
    
    # build dataset
    mnist_dataset_train = paddle.dataset.mnist.train()
    mnist_dataset_test = paddle.dataset.mnist.test()

    # configure the network

    image_encoder1 = ImageMLPEncoder('model1')
    image_encoder1.model['hidden_layer_2'] = image_encoder1.model['hidden_layer_1']
    
    image_encoder1.build()

    loss = fluid.layers.cross_entropy(input=image_encoder1.prediction, label=image_encoder1.label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.RMSProp(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    fetch_params = [avg_loss]

    # test fetching gradients via parameters
    params = image_encoder1.get_param('hidden_layer_1')
    for _var_name in params:
        for param in params[_var_name]:
            grads = image_encoder1.get_gradient(param)
            for gard_var_name in grads:
                for grad_vars in grads[gard_var_name]:
                    print(gard_var_name, grad_vars.name)
                    fetch_params.append(grad_vars)

    # build reader
    train_reader = paddle.batch(paddle.reader.shuffle(mnist_dataset_train, buf_size=500), batch_size=128)

    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[image_encoder1.input, image_encoder1.label], place=place)

    # prepare running
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program()) 

    for epoch in range(3):
        for step_id, data in enumerate(train_reader()):
            output = exe.run(fluid.default_main_program(), feed=feeder.feed(data), fetch_list=fetch_params)
            if step_id % 100 == 0:
                print("training step %d, training loss %.3f" % (step_id, output[0]))
            if step_id % 300 == 0:
                for i in range(1, len(output)):
                    print(output[i])


```



```python
import os
import sys
import paddle
import paddle.fluid as fluid

from paddlecraft.image_mlp_encoder import ImageMLPEncoder


if __name__ == "__main__":
    
    # build dataset
    mnist_dataset_train = paddle.dataset.mnist.train()
    mnist_dataset_test = paddle.dataset.mnist.test()

    # configure the network

    image_encoder1 = ImageMLPEncoder('model1')
    image_encoder1.model['hidden_layer_2'] = image_encoder1.model['hidden_layer_1']
    
    image_encoder1.build()

    loss = fluid.layers.cross_entropy(input=image_encoder1.prediction, label=image_encoder1.label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.RMSProp(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    fetch_params = [avg_loss]

    # test fetching gradient
    print("="*50 + "show gradients" + "="*50)
    grads = image_encoder1.get_gradient('prediction')
    for gard_var_name in grads:
        for grad_vars in grads[gard_var_name]:
            print(gard_var_name, grad_vars.name)
            fetch_params.append(grad_vars)

    # build reader
    train_reader = paddle.batch(paddle.reader.shuffle(mnist_dataset_train, buf_size=500), batch_size=128)

    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[image_encoder1.input, image_encoder1.label], place=place)

    # prepare running
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program()) 

    for epoch in range(3):
        for step_id, data in enumerate(train_reader()):
            output = exe.run(fluid.default_main_program(), feed=feeder.feed(data), fetch_list=fetch_params)
            if step_id % 100 == 0:
                print("training step %d, training loss %.3f" % (step_id, output[0]))
            if step_id % 300 == 0:
                for i in range(1, len(output)):
                    print(output[i])


```



case5：冻结/解冻

```python
import os
import sys
import paddle
import paddle.fluid as fluid

from paddlecraft.image_mlp_encoder import ImageMLPEncoder


if __name__ == "__main__":
    
    # build dataset
    mnist_dataset_train = paddle.dataset.mnist.train()
    mnist_dataset_test = paddle.dataset.mnist.test()

    # configure the network

    image_encoder1 = ImageMLPEncoder('model1')

    with fluid.program_guard(fluid.default_main_program()):
        with fluid.unique_name.guard():
            image_encoder1.build()
            image_encoder1.prediction.stop_gradient = True

            loss = fluid.layers.cross_entropy(input=image_encoder1.prediction, label=image_encoder1.label)
            avg_loss = fluid.layers.mean(loss)

            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
            optimizer.minimize(avg_loss)

    # build reader
    train_reader = paddle.batch(paddle.reader.shuffle(mnist_dataset_train, buf_size=500), batch_size=128)

    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[image_encoder1.input, image_encoder1.label], place=place)

    # prepare running
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program()) 

    for epoch in range(3):
        for step_id, data in enumerate(train_reader()):
            output = exe.run(fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_loss])
            if step_id % 100 == 0:
                print("training step %d, training loss %.3f" % (step_id, output[0]))
            if step_id % 300 == 0:
                for i in range(1, len(output)):
                    print(output[i])

    # unfreeze the model

    train_program = fluid.Program()
    fluid.framework.switch_main_program(train_program)

    with fluid.program_guard(fluid.default_main_program()):
        with fluid.unique_name.guard():
            image_encoder1.build()
            image_encoder1.prediction.stop_gradient = False

            loss = fluid.layers.cross_entropy(input=image_encoder1.prediction, label=image_encoder1.label)
            avg_loss = fluid.layers.mean(loss)

            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
            optimizer.minimize(avg_loss)

    for epoch in range(3):
        for step_id, data in enumerate(train_reader()):
            output = exe.run(fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_loss])
            if step_id % 100 == 0:
                print("training step %d, training loss %.3f" % (step_id, output[0]))
            if step_id % 300 == 0:
                for i in range(1, len(output)):
                    print(output[i])

```





Case7: image + langage