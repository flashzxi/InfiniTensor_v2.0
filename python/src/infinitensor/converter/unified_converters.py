import torch.nn as nn
from .registry import registry

#https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml

@registry.register("matmul","default")
def convert_matmul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.gemm(a, b, None)

@registry.register("add","Tensor")
def convert_add(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.add(a, b, None)

@registry.register("mul","Tensor")
def convert_mul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.mul(a, b, None)

@registry.register("sub","Tensor")
def convert_sub(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.sub(a, b, None)

@registry.register("clamp","Tensor")
def convert_clamp(translator, node):
    t = translator.tensors[node.args[0]]
    mmin = translator.tensors[node.args[1]]
    mmax = translator.tensors[node.args[2]]
    translator.tensors[node] = translator.builder.clamp(t, mmin, mmax, None)