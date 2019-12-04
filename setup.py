# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2019  Baidu.com, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
################################################################################
"""
Setup script.
Authors: zhouxiangyang(zhouxiangyang@baidu.com)
Date:    2019/09/29 21:00:01
"""
import io
import setuptools

setuptools.setup(
    name="paddlecraft",
    version="0.0.1",
    author="PaddlePaddle",
    author_email="zhouxiangyang@baidu.com",
    description="Make Neural Models as APIs for solving more complicated AI problems",
    url="https://github.com/PaddlePadd",
    # packages=setuptools.find_packages(),
    packages=['paddlecraft', 'paddlecraft.utils'],
    package_dir={
        'paddlecraft': './paddlecraft',
        'paddlecraft.utils': './paddlecraft/utils'
    },
    platforms="any",
    license='Apache 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ], )
