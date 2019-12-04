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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid

#from model.bert import BertModel
from paddlecraft.bert import Bert


def create_model(bert_config, num_labels, is_prediction=False):
    input_fields = {
        'names': ['src_ids', 'pos_ids', 'sent_ids', 'input_mask', 'labels'],
        'shapes': [[None, None], [None, None], [None, None], [None, None, 1],
                   [None, 1]],
        'dtypes': ['int64', 'int64', 'int64', 'float32', 'int64'],
        'lod_levels': [0, 0, 0, 0, 0],
    }

    inputs = [
        fluid.data(
            name=input_fields['names'][i],
            shape=input_fields['shapes'][i],
            dtype=input_fields['dtypes'][i],
            lod_level=input_fields['lod_levels'][i])
        for i in range(len(input_fields['names']))
    ]
    (src_ids, pos_ids, sent_ids, input_mask, labels) = inputs

    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=inputs, capacity=50, iterable=False)

    bert = Bert(
        name='encoder',
        n_layer=bert_config.n_layer,
        n_head=bert_config.n_head,
        d_wordembedding=bert_config.d_wordembedding,
        d_key=bert_config.d_wordembedding // bert_config.n_head,
        d_value=bert_config.d_wordembedding // bert_config.n_head,
        d_model=bert_config.d_model,
        d_inner_hid=bert_config.d_model * 4,
        prepostprocess_dropout=bert_config.hidden_dropout_prob,
        attention_dropout=bert_config.attention_probs_dropout_prob,
        relu_dropout=0.,
        hidden_act=bert_config.hidden_act,
        preprocess_cmd="",
        postprocess_cmd="dan",
        param_initializer=fluid.initializer.TruncatedNormal(scale=0.02),
        is_training=True,
        is_pretraining=False,
        weight_sharing=True,
        vocab_size=bert_config.vocab_size,
        max_position_seq_len=bert_config.max_position_embeddings,
        sent_types=bert_config.type_vocab_size, )

    bert.build(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask)

    cls_feats = bert.get_pooled_output()

    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")

    logits = fluid.layers.fc(
        input=cls_feats,
        num_flatten_dims=2,
        size=num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    if is_prediction:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, input_mask.name
        ]
        return data_loader, probs, feed_targets_name

    logits = fluid.layers.reshape(logits, [-1, num_labels], inplace=True)
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    return data_loader, loss, probs, accuracy, num_seqs


if __name__ == "__main__":
    print("hello world!")
