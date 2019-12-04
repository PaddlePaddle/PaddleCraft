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
"""BERT in PaddleCraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import paddle
import paddle.fluid as fluid

from paddlecraft.base import logical2physical_name, physical2logical_name
from paddlecraft import TransformerEncoder


class Bert(TransformerEncoder):
    def __init__(
            self,
            name,
            n_layer=12,
            n_head=12,
            d_wordembedding=768,
            d_key=64,
            d_value=64,
            d_model=768,
            d_inner_hid=768 * 4,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.,
            hidden_act="gelu",
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=fluid.initializer.TruncatedNormal(scale=0.02),
            is_training=True,
            is_pretraining=False,
            weight_sharing=True,
            vocab_size=30000,
            max_position_seq_len=64,
            sent_types=2):

        super(Bert, self).__init__(
            name,
            n_layer=n_layer,
            n_head=n_head,
            d_wordembedding=d_wordembedding,
            d_key=d_key,
            d_value=d_value,
            d_model=d_model,
            d_inner_hid=d_inner_hid,
            prepostprocess_dropout=prepostprocess_dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            hidden_act=hidden_act,
            preprocess_cmd=preprocess_cmd,
            postprocess_cmd=postprocess_cmd,
            param_initializer=param_initializer,
            is_training=is_training)

        self.weight_sharing = weight_sharing
        self.vocab_size = vocab_size
        self.max_position_seq_len = max_position_seq_len
        self.sent_types = sent_types
        self.is_pretraining = is_pretraining

        self.param_names['input'] = {
            'word_embedding':
            logical2physical_name(self.name, 'word_embedding'),
            'pos_embedding': logical2physical_name(self.name, 'pos_embedding'),
            'sent_embedding':
            logical2physical_name(self.name, 'sent_embedding'),
            'layer_norm_scale':
            logical2physical_name(self.name, 'pre_encoder_layer_norm_scale'),
            'layer_norm_bias':
            logical2physical_name(self.name, 'pre_encoder_layer_norm_bias'),
        }

        self.param_names['masked_language_model'] = {
            'trans_fc': {
                'w': logical2physical_name(self.name, 'mask_lm_trans_fc.w_0'),
                'b': logical2physical_name(self.name, 'mask_lm_trans_fc.b_0'),
            },
            'trans_layer_norm': {
                'scale': logical2physical_name(
                    self.name, 'mask_lm_trans_layer_norm_scale'),
                'bias': logical2physical_name(self.name,
                                              'mask_lm_trans_layer_norm_bias'),
            },
            'out': {
                'w': logical2physical_name(self.name, 'mask_lm_out_fc.w_0'),
                'b': logical2physical_name(self.name, 'mask_lm_out_fc.b_0'),
            }
        } if self.is_pretraining else None

        self.param_names['next_sentence_prediction'] = {
            'pooled_fc': {
                'w': logical2physical_name(self.name, 'pooled_fc.w_0'),
                'b': logical2physical_name(self.name, 'pooled_fc.b_0')
            },
            'out_fc': {
                'w': logical2physical_name(self.name, 'next_sent_fc.w_0'),
                'b': logical2physical_name(self.name, 'next_sent_fc.b_0'),
            }
        }

        self.model['input'] = {
            'word_embedding': None,
            'pos_embedding': None,
            'sent_embedding': None
        }

    def build(self,
              src_ids,
              position_ids,
              sentence_ids,
              input_mask,
              mask_label=None,
              mask_pos=None,
              labels=None):

        if not self.is_pretraining:
            assert mask_label == None
            assert mask_pos == None
            assert labels == None

        if self.is_pretraining:
            assert mask_label != None
            assert mask_pos != None
            assert labels != None

        emb_out = fluid.embedding(
            input=src_ids,
            size=[self.vocab_size, self.d_wordembedding],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name=self.param_names['input']['word_embedding'],
                initializer=self.param_initializer),
            is_sparse=False)

        position_emb_out = fluid.embedding(
            input=position_ids,
            size=[self.max_position_seq_len, self.d_wordembedding],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name=self.param_names['input']['pos_embedding'],
                initializer=self.param_initializer))

        sent_emb_out = fluid.embedding(
            sentence_ids,
            size=[self.sent_types, self.d_wordembedding],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name=self.param_names['input']['sent_embedding'],
                initializer=self.param_initializer))

        self.model['input']['word_embedding'] = emb_out
        self.model['input']['pos_embedding'] = position_emb_out
        self.model['input']['sent_embedding'] = sent_emb_out

        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out

        emb_out = self.pre_process_layer(
            emb_out,
            'nd',
            self.prepostprocess_dropout,
            scale_name=self.param_names['input']['layer_norm_scale'],
            bias_name=self.param_names['input']['layer_norm_bias'])

        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)
        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self.n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        super(Bert, self).build(emb_out, n_head_self_attn_mask)

        if self.is_pretraining:
            self.get_pretraining_output(mask_label, mask_pos, labels)

    def get_sequence_output(self):
        return self.model['out']

    def get_pooled_output(self):
        next_sent_feat = fluid.layers.slice(
            input=self.model['out'], axes=[1], starts=[0], ends=[1])

        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            num_flatten_dims=2,
            size=self.d_wordembedding,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name=self.param_names['next_sentence_prediction']['pooled_fc'][
                    'w'],
                initializer=self.param_initializer),
            bias_attr=self.param_names['next_sentence_prediction']['pooled_fc'][
                'b'])

        return next_sent_feat

    def get_pretraining_output(self, mask_label, mask_pos, labels):
        """Get the loss & accuracy for pretraining"""

        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # extract the first token feature in each sentence
        next_sent_feat = self.get_pooled_output()
        reshaped_emb_out = fluid.layers.reshape(
            x=self.model['out'], shape=[-1, self.d_wordembedding])

        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # transform: fc
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self.d_wordembedding,
            act=self.hidden_act,
            param_attr=fluid.ParamAttr(
                name=self.param_names['masked_language_model']['trans_fc']['w'],
                initializer=self.param_initializer),
            bias_attr=fluid.ParamAttr(name=self.param_names[
                'masked_language_model']['trans_fc']['b']))

        # transform: layer norm
        mask_trans_feat = self.pre_process_layer(
            mask_trans_feat,
            'n',
            scale_name=self.param_names['masked_language_model'][
                'trans_layer_norm']['scale'],
            bias_name=self.param_names['masked_language_model'][
                'trans_layer_norm']['bias'])

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name=self.param_names['masked_language_model']['out']['b'],
            initializer=fluid.initializer.Constant(value=0.0))

        if self.weight_sharing:
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self.param_names['input']['word_embedding']),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self.vocab_size],
                dtype='float32',
                attr=mask_lm_out_bias_attr,
                is_bias=True)
        else:
            fc_out = fluid.layers.fc(
                input=mask_trans_feat,
                size=self.vocab_size,
                param_attr=fluid.ParamAttr(
                    name=self.param_names['masked_language_model']['out']['w'],
                    initializer=self.param_initializer),
                bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)

        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)

        next_sent_fc_out = fluid.layers.fc(
            input=next_sent_feat,
            num_flatten_dims=2,
            size=2,
            param_attr=fluid.ParamAttr(
                name=self.param_names['next_sentence_prediction']['out_fc'][
                    'w'],
                initializer=self.param_initializer),
            bias_attr=self.param_names['next_sentence_prediction']['out_fc'][
                'b'])

        next_sent_fc_out = fluid.layers.reshape(
            next_sent_fc_out, [-1, 2], inplace=True)
        next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=next_sent_fc_out, label=labels, return_softmax=True)

        next_sent_acc = fluid.layers.accuracy(
            input=next_sent_softmax, label=labels)

        mean_next_sent_loss = fluid.layers.mean(next_sent_loss)

        loss = mean_next_sent_loss + mean_mask_lm_loss

        self.model['loss'] = {
            'loss': loss,
            'next_sentence_prediction_loss': mean_next_sent_loss,
            'masked_language_model_loss': mean_mask_lm_loss,
        }

        self.model['acc'] = {'next_sentence_acc': next_sent_acc}

        return next_sent_acc, mean_mask_lm_loss, loss
