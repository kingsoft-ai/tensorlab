# ----------------------------------------------------------------------------
# Copyright 2017 King soft.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import os
import sys
import shutil
from setuptools import Extension, setup, find_packages, Command


# Define version information
VERSION = '0.1'
FULLVERSION = VERSION


# build python

setup(name='tensorlab',
    version=VERSION,
    description="tensorflow framework",
    #long_description=open('README.md').read(),
    author='yi gu',
    author_email='390512308@qq.com',
    license='License :: OSI Approved :: Apache Software License',
    packages=find_packages(),
    #package_dir={'': 'tensorlab'},
    package_data={'tensorlab': ['*.so']},
    zip_safe=False,
    install_requires = [
        'tensorflow>=1.0',
        'numpy>=1.13.1',
        'pyyaml>=3.12',
        'image>=1.5.16',
        'easydict>=1.7',
        'opencv-python>=3.3.0.10',
        'progressbar2>=3.34.2',
    ],
    classifiers=['Development Status :: 3 - Alpha',
               'Environment :: Console',
               'Environment :: Console :: Curses',
               'Environment :: Web Environment',
               'Intended Audience :: End Users/Desktop',
               'Intended Audience :: Developers',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: Apache Software License',
               'Operating System :: POSIX',
               'Operating System :: MacOS :: MacOS X',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: ' +
               'Artificial Intelligence',
               'Topic :: Scientific/Engineering :: Information Analysis',
               'Topic :: System :: Distributed Computing'])