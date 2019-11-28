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

