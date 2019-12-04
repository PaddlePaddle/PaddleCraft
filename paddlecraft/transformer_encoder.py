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
"""Transformer encoder in PaddleCraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from paddlecraft.base import logical2physical_name, physical2logical_name
from paddlecraft.base import BaseModel


class TransformerEncoder(BaseModel):
    def __init__(
            self,
            name,
            n_layer=12,
            n_head=12,
            d_wordembedding=768,
            d_key=64,  # embedding_size // num_heads
            d_value=64,  # embedding_size // num_heads
            d_model=768,
            d_inner_hid=768 * 4,  # embedding_size * 4
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.,
            hidden_act="gelu",
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=fluid.initializer.TruncatedNormal(scale=0.02),
            is_training=True):

        self.name = name
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_wordembedding = d_wordembedding
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.prepostprocess_dropout = prepostprocess_dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.hidden_act = hidden_act
        self.preprocess_cmd = preprocess_cmd
        self.postprocess_cmd = postprocess_cmd
        self.param_initializer = param_initializer
        self.is_training = is_training

        self.param_names = {'blocks': []}

        self.model = {'blocks': []}

        # define the structure of transformer_encoder
        # the self.param_names can also help debug and print the model strcuture.
        for i in range(self.n_layer):
            curr_block = {}
            curr_block_physical_name = logical2physical_name(self.name,
                                                             'layer_' + str(i))

            curr_model_block = {
                'multi_head_attention': {
                    'softmax': None,
                    'fusion': None,
                    'out': None,
                },
                'feedforward': {
                    'fusion': None,
                    'out': None
                }
            }

            curr_block['multi_head_attention'] = {
                'pre_attention': {
                    'scale': logical2physical_name(curr_block_physical_name,
                                                   'pre_att_layer_norm_scale'),
                    'bias': logical2physical_name(curr_block_physical_name,
                                                  'pre_att_layer_norm_bias'),
                } if 'n' in self.preprocess_cmd else {},
                'attention_fc': {
                    'query': {
                        'w':
                        logical2physical_name(curr_block_physical_name,
                                              'multi_head_att_query_fc.w_0'),
                        'b':
                        logical2physical_name(curr_block_physical_name,
                                              'multi_head_att_query_fc.b_0')
                    },
                    'key': {
                        'w': logical2physical_name(curr_block_physical_name,
                                                   'multi_head_att_key_fc.w_0'),
                        'b': logical2physical_name(curr_block_physical_name,
                                                   'multi_head_att_key_fc.b_0')
                    },
                    'value': {
                        'w':
                        logical2physical_name(curr_block_physical_name,
                                              'multi_head_att_value_fc.w_0'),
                        'b':
                        logical2physical_name(curr_block_physical_name,
                                              'multi_head_att_value_fc.b_0')
                    },
                    'output': {
                        'w':
                        logical2physical_name(curr_block_physical_name,
                                              'multi_head_att_output_fc.w_0'),
                        'b':
                        logical2physical_name(curr_block_physical_name,
                                              'multi_head_att_output_fc.b_0')
                    },
                },
                'post_attention': {
                    'scale': logical2physical_name(curr_block_physical_name,
                                                   'post_att_layer_norm_scale'),
                    'bias': logical2physical_name(curr_block_physical_name,
                                                  'post_att_layer_norm_bias')
                } if 'n' in self.postprocess_cmd else {}
            }
            curr_block['feedforward'] = {
                'pre_feedforward': {
                    'scale': logical2physical_name(curr_block_physical_name,
                                                   'pre_ffn_layer_norm_scale'),
                    'bias': logical2physical_name(curr_block_physical_name,
                                                  'pre_ffn_layer_norm_bias'),
                } if 'n' in self.preprocess_cmd else {},
                'feedforward_fc': [{
                    'w': logical2physical_name(curr_block_physical_name,
                                               'ffn_fc_0.w_0'),
                    'b': logical2physical_name(curr_block_physical_name,
                                               'ffn_fc_0.b_0'),
                }, {
                    'w': logical2physical_name(curr_block_physical_name,
                                               'ffn_fc_1.w_0'),
                    'b': logical2physical_name(curr_block_physical_name,
                                               'ffn_fc_1.b_0'),
                }],
                'post_feedforward': {
                    'scale': logical2physical_name(curr_block_physical_name,
                                                   'post_ffn_layer_norm_scale'),
                    'bias': logical2physical_name(curr_block_physical_name,
                                                  'post_ffn_layer_norm_bias'),
                } if 'n' in self.postprocess_cmd else {}
            }

            self.param_names['blocks'].append(curr_block)
            self.model['blocks'].append(curr_model_block)

        self.param_names['post_encoder'] = {
            'scale':
            logical2physical_name(self.name, 'post_encoder_layer_norm_scale'),
            'bias':
            logical2physical_name(self.name, 'post_encoder_layer_norm_bias'),
        }
        self.model['out'] = None

    def multi_head_attention(self,
                             queries,
                             keys,
                             values,
                             attn_bias,
                             curr_block_id,
                             d_key,
                             d_value,
                             d_model,
                             n_head=1,
                             dropout_rate=0.,
                             cache=None,
                             param_initializer=None):
        """
        Multi-Head Attention. Note that attn_bias is added to the logit before
        computing softmax activiation to mask certain selected positions so that
        they will not considered in attention weights.
        """

        curr_param_block = self.param_names['blocks'][curr_block_id]

        keys = queries if keys is None else keys
        values = keys if values is None else values

        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3
                ):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")

        def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
            """
            Add linear projection to queries, keys, and values.
            """
            q = layers.fc(input=queries,
                          size=d_key * n_head,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=curr_param_block['multi_head_attention'][
                                  'attention_fc']['query']['w'],
                              initializer=param_initializer),
                          bias_attr=curr_param_block['multi_head_attention'][
                              'attention_fc']['query']['b'])
            k = layers.fc(input=keys,
                          size=d_key * n_head,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=curr_param_block['multi_head_attention'][
                                  'attention_fc']['key']['w'],
                              initializer=param_initializer),
                          bias_attr=curr_param_block['multi_head_attention'][
                              'attention_fc']['key']['b'])
            v = layers.fc(input=values,
                          size=d_value * n_head,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=curr_param_block['multi_head_attention'][
                                  'attention_fc']['value']['w'],
                              initializer=param_initializer),
                          bias_attr=curr_param_block['multi_head_attention'][
                              'attention_fc']['value']['b'])
            return q, k, v

        def __split_heads(x, n_head):
            """
            Reshape the last dimension of inpunt tensor x so that it becomes two
            dimensions and then transpose. Specifically, input a tensor with shape
            [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
            with shape [bs, n_head, max_sequence_length, hidden_dim].
            """
            hidden_size = x.shape[-1]
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            reshaped = layers.reshape(
                x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

            # permuate the dimensions into:
            # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
            return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

        def __combine_heads(x):
            """
            Transpose and then reshape the last two dimensions of inpunt tensor x
            so that it becomes one dimension, which is reverse to __split_heads.
            """
            if len(x.shape) == 3: return x
            if len(x.shape) != 4:
                raise ValueError("Input(x) should be a 4-D Tensor.")

            trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            return layers.reshape(
                x=trans_x,
                shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
                inplace=True)

        def scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                         dropout_rate):
            """
            Scaled Dot-Product Attention
            """
            scaled_q = layers.scale(x=q, scale=d_key**-0.5)
            product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
            if attn_bias:
                product += attn_bias
            weights = layers.softmax(product)

            # memorize the weights in block
            self.model['blocks'][curr_block_id]['multi_head_attention'][
                'softmax'] = weights

            if dropout_rate and self.is_training:
                weights = layers.dropout(
                    weights,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)

            out = layers.matmul(weights, v)
            return out

        q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

        if cache is not None:  # use cache and concat time steps
            # Since the inplace reshape in __split_heads changes the shape of k and
            # v, which is the cache input for next time step, reshape the cache
            # input from the previous time step first.
            k = cache["k"] = layers.concat(
                [layers.reshape(
                    cache["k"], shape=[0, 0, d_model]), k], axis=1)
            v = cache["v"] = layers.concat(
                [layers.reshape(
                    cache["v"], shape=[0, 0, d_model]), v], axis=1)

        q = __split_heads(q, n_head)
        k = __split_heads(k, n_head)
        v = __split_heads(v, n_head)

        ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                      dropout_rate)

        out = __combine_heads(ctx_multiheads)

        # Project back to the model size.
        proj_out = layers.fc(input=out,
                             size=d_model,
                             num_flatten_dims=2,
                             param_attr=fluid.ParamAttr(
                                 name=curr_param_block['multi_head_attention'][
                                     'attention_fc']['output']['w'],
                                 initializer=param_initializer),
                             bias_attr=curr_param_block['multi_head_attention'][
                                 'attention_fc']['output']['b'])

        return proj_out

    def positionwise_feed_forward(self,
                                  x,
                                  curr_block_id,
                                  d_inner_hid,
                                  d_hid,
                                  dropout_rate,
                                  hidden_act,
                                  param_initializer=None):
        """
        Position-wise Feed-Forward Networks.
        This module consists of two linear transformations with a ReLU activation
        in between, which is applied to each position separately and identically.
        """

        curr_param_block = self.param_names['blocks'][curr_block_id]

        hidden = layers.fc(
            input=x,
            size=d_inner_hid,
            num_flatten_dims=2,
            act=hidden_act,
            param_attr=fluid.ParamAttr(
                name=curr_param_block['feedforward']['feedforward_fc'][0]['w'],
                initializer=param_initializer),
            bias_attr=curr_param_block['feedforward']['feedforward_fc'][0]['b'])

        if dropout_rate and self.is_training:
            hidden = layers.dropout(
                hidden,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)

        out = layers.fc(
            input=hidden,
            size=d_hid,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name=curr_param_block['feedforward']['feedforward_fc'][1]['w'],
                initializer=param_initializer),
            bias_attr=curr_param_block['feedforward']['feedforward_fc'][1]['b'])

        return out

    def pre_post_process_layer(self,
                               prev_out,
                               out,
                               process_cmd,
                               dropout_rate=0.,
                               scale_name='',
                               bias_name='',
                               inner_res={}):
        """
        Add residual connection, layer normalization and droput to the out tensor
        optionally according to the value of process_cmd.
        This will be used before or after multi-head attention and position-wise
        feed-forward networks.
        """

        for cmd in process_cmd:
            if cmd == "a":  # add residual connection
                out = out + prev_out if prev_out else out
                inner_res['a'] = out
            elif cmd == "n":  # add layer normalization
                out_dtype = out.dtype
                if out_dtype == fluid.core.VarDesc.VarType.FP16:
                    out = layers.cast(x=out, dtype="float32")
                out = layers.layer_norm(
                    out,
                    begin_norm_axis=len(out.shape) - 1,
                    param_attr=fluid.ParamAttr(
                        name=scale_name,
                        initializer=fluid.initializer.Constant(1.)),
                    bias_attr=fluid.ParamAttr(
                        name=bias_name,
                        initializer=fluid.initializer.Constant(0.)))
                if out_dtype == fluid.core.VarDesc.VarType.FP16:
                    out = layers.cast(x=out, dtype="float16")
                inner_res['n'] = out
            elif cmd == "d":  # add dropout
                if dropout_rate and self.is_training:
                    out = layers.dropout(
                        out,
                        dropout_prob=dropout_rate,
                        dropout_implementation="upscale_in_train",
                        is_test=False)
                    inner_res['d'] = out
        return out

    def pre_process_layer(self,
                          out,
                          process_cmd,
                          dropout_rate=0.,
                          scale_name='',
                          bias_name='',
                          inner_res={}):

        final_out = self.pre_post_process_layer(
            None,
            out,
            process_cmd,
            dropout_rate,
            scale_name=scale_name,
            bias_name=bias_name,
            inner_res=inner_res)

        return final_out

    def post_process_layer(self,
                           prev_out,
                           out,
                           process_cmd,
                           dropout_rate=0.,
                           scale_name='',
                           bias_name='',
                           inner_res={}):

        return self.pre_post_process_layer(
            prev_out,
            out,
            process_cmd,
            dropout_rate,
            scale_name=scale_name,
            bias_name=bias_name,
            inner_res=inner_res)

    def encoder_layer(self,
                      enc_input,
                      attn_bias,
                      curr_block_id,
                      n_head,
                      d_key,
                      d_value,
                      d_model,
                      d_inner_hid,
                      prepostprocess_dropout,
                      attention_dropout,
                      relu_dropout,
                      hidden_act,
                      preprocess_cmd="n",
                      postprocess_cmd="da",
                      param_initializer=None):
        """The encoder layers that can be stacked to form a deep encoder.
        This module consits of a multi-head (self) attention followed by
        position-wise feed-forward networks and both the two components companied
        with the post_process_layer to add residual connection, layer normalization
        and droput.
        """

        curr_param_names = self.param_names['blocks'][curr_block_id]
        _inner_res = {}

        attn_output = self.multi_head_attention(
            self.pre_process_layer(
                enc_input,
                preprocess_cmd,
                prepostprocess_dropout,
                scale_name='' if 'scale' not in
                curr_param_names['multi_head_attention']['pre_attention'] else
                curr_param_names['multi_head_attention']['pre_attention'][
                    'scale'],
                bias_name='' if 'bias' not in
                curr_param_names['multi_head_attention']['pre_attention'] else
                curr_param_names['multi_head_attention']['pre_attention'][
                    'bias']),
            None,
            None,
            attn_bias,
            curr_block_id,
            d_key,
            d_value,
            d_model,
            n_head,
            attention_dropout,
            param_initializer=param_initializer)

        attn_output = self.post_process_layer(
            enc_input,
            attn_output,
            postprocess_cmd,
            prepostprocess_dropout,
            scale_name='' if 'scale' not in
            curr_param_names['multi_head_attention']['post_attention'] else
            curr_param_names['multi_head_attention']['post_attention']['scale'],
            bias_name='' if 'bias' not in
            curr_param_names['multi_head_attention']['post_attention'] else
            curr_param_names['multi_head_attention']['post_attention']['bias'],
            inner_res=_inner_res)

        if 'a' in _inner_res:
            self.model['blocks'][curr_block_id]['multi_head_attention'][
                'fusion'] = _inner_res['a']
        if 'd' in _inner_res and 'a' in _inner_res:
            self.model['blocks'][curr_block_id]['multi_head_attention'][
                'out'] = _inner_res['n']

        ffd_output = self.positionwise_feed_forward(
            self.pre_process_layer(
                attn_output,
                preprocess_cmd,
                prepostprocess_dropout,
                scale_name='' if 'scale' not in
                curr_param_names['feedforward']['pre_feedforward'] else
                curr_param_names['feedforward']['pre_feedforward']['scale'],
                bias_name='' if
                'bias' not in curr_param_names['feedforward']['pre_feedforward']
                else
                curr_param_names['feedforward']['pre_feedforward']['bias']),
            curr_block_id,
            d_inner_hid,
            d_model,
            relu_dropout,
            hidden_act,
            param_initializer=param_initializer)

        _inner_res.clear()
        final_out = self.post_process_layer(
            attn_output,
            ffd_output,
            postprocess_cmd,
            prepostprocess_dropout,
            scale_name='' if
            'scale' not in curr_param_names['feedforward']['post_feedforward']
            else curr_param_names['feedforward']['post_feedforward']['scale'],
            bias_name=''
            if 'bias' not in curr_param_names['feedforward']['post_feedforward']
            else curr_param_names['feedforward']['post_feedforward']['bias'],
            inner_res=_inner_res)

        if 'a' in _inner_res:
            self.model['blocks'][curr_block_id]['feedforward'][
                'fusion'] = _inner_res['a']

        return final_out

    def build(
            self,
            enc_input,
            attn_bias, ):
        """
        The encoder is composed of a stack of identical layers returned by calling
        encoder_layer.
        """
        for i in range(self.n_layer):
            enc_output = self.encoder_layer(
                enc_input,
                attn_bias,
                curr_block_id=i,
                n_head=self.n_head,
                d_key=self.d_key,
                d_value=self.d_value,
                d_model=self.d_model,
                d_inner_hid=self.d_inner_hid,
                prepostprocess_dropout=self.prepostprocess_dropout,
                attention_dropout=self.attention_dropout,
                relu_dropout=self.relu_dropout,
                hidden_act=self.hidden_act,
                preprocess_cmd=self.preprocess_cmd,
                postprocess_cmd=self.postprocess_cmd,
                param_initializer=self.param_initializer)
            self.model['blocks'][i]['feedforward']['out'] = enc_output
            enc_input = enc_output

        enc_output = self.pre_process_layer(
            enc_output,
            self.preprocess_cmd,
            self.prepostprocess_dropout,
            scale_name=self.param_names['post_encoder']['scale'],
            bias_name=self.param_names['post_encoder']['bias'])

        self.model['out'] = enc_output

        return enc_output


if __name__ == "__main__":
    bert = TransformerEncoder('bert_base')

    embedding_input = fluid.data(
        shape=[-1, 128, 764], dtype='float32', name='embedding')
    input_mask = fluid.data(shape=[-1, 128, 1], dtype='float32', name='mask')
    att_mask = fluid.layers.matmul(x=input_mask, y=input_mask, transpose_y=True)

    bert.build(embedding_input, att_mask)

    print(bert.model)
