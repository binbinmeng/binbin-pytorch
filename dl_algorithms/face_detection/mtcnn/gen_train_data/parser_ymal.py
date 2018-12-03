#!/usr/bin/python
# -*- coding: utf-8 -*-

import yaml
f = open('config.yaml')
x = yaml.load(f)
print(x['MODEL_STORE_DIR'])
