"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    # TODO: Multiply together the elements of a shape tuple to get the total number of elements.
    if len(shape) == 0:
        return 1 
    
    if len(shape) == 1:
        if isinstance(shape , tuple):
            return shape
        elif isinstance(shape, list):
            return shape[0]
        else:
            return 1

    n = 1
    n = [n := n * i for i in shape]
    return n[-1]

# Step 2 - argsort
def argsort(values):
    # TODO: Return the indices that would sort values in ascending order.
    return sorted(range(len(values)), key=lambda i: values[i])

# Step 3 - make_op_enums
from enum import Enum

class Unary(Enum):
    NEG = 1
    RELU = 2
    LOG = 3
    EXP = 4
    SQRT = 5
    SIGMOID = 6

class Binary(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    CMPLT = 5
    MAX = 6

class Reduce(Enum):
    SUM = 1
    MAX = 2

class Movement(Enum):
    RESHAPE = 1
    EXPAND = 2
    PERMUTE = 3

def make_op_enums():
    return Unary, Binary, Reduce, Movement

# Step 4 - LazyBuffer
import numpy as np

class LazyBuffer:
    def __init__(self, np_array):
        self._np = np.asarray(np_array)
        self.shape = tuple(int(d) for d in self._np.shape)
        self.dtype = self._np.dtype

    def  __array__(self, dtype):
        return np.asarray(self._np, dtype=dtype)

    def __float__(self):
        return float(self._np)

# Step 5 - lazybuffer_const
import numpy as np

def const(value, shape):
    np_array = np.full(shape, value, dtype=np.float32)
    return LazyBuffer(np_array)

LazyBuffer.const = staticmethod(const)

# Step 6 - rand
import numpy as np

def rand(shape, seed=None):
    rng = np.random.RandomState(seed)
    rnd_arr = rng.random(shape).astype(np.float32)
    return LazyBuffer(rnd_arr)

# Step 7 - lazybuffer_unary_e
import numpy as np


def e(self, op):
    x = self._np
    if op.name == "NEG":
        x = -x
    elif op.name == "RELU":
        x = np.maximum(0, x)
    elif op.name == "LOG":
        x = np.log(x, dtype=self.dtype)
    elif op.name == "EXP":
        x = np.exp(x, dtype=self.dtype)
    elif op.name == "SQRT":
        x = np.sqrt(x, dtype=self.dtype)
    elif op.name == "SIGMOID":
        x = 1 / (1 + np.exp(-x, dtype=self.dtype))
    else:
        raise ValueError("Unknown ops.")
    
    return LazyBuffer(x)

LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self, op, other):
    a = self._np
    b = other._np 

    if op.name == "ADD":
        a = a + b 
    elif op.name == "SUB":
        a = a - b 
    elif op.name == "MUL":
        a = a * b 
    elif op.name == "DIV":
        a = a / b 
    elif op.name == "CMPLT":
        a = np.where(a < b, 1, 0).astype(a.dtype)
    elif op.name == "MAX":
        a = np.maximum(a, b).astype(a.dtype)
    else:
         raise ValueError("Unknown ops.")
    
    return LazyBuffer(a)

# Step 9 - lazybuffer_r
def r(self, op, axis):
    a = self._np
    name = getattr(op, 'name', None)
    if name == "SUM":
        out = np.sum(a, axis, keepdims=True)
    elif name == "MAX":
        out = np.max(a, axis, keepdims=True)
    else:
        raise ValueError("Unknown ops.")

    return LazyBuffer(out)

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    # TODO: return a new LazyBuffer with the array reshaped to new_shape
    a = self._np 
    out = a.reshape(new_shape)
    return LazyBuffer(out)

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    # TODO: broadcast this buffer's size-1 dims out to new_shape
    normilized_shape = (int(d) for d in new_shape)
    a = self._np 
    try:
        views = np.broadcast_to(a, normilized_shape)
        out = np.array(views, dtype=a.dtype)
        return LazyBuffer(out)
    except:
        raise ValueError("Shapes are incompatible.")

# Step 12 - lazybuffer_permute
def permute(self, order):
    # TODO: return a new LazyBuffer with axes reordered according to order
    a = self._np 
    try:
        out = np.transpose(a, order)
        return LazyBuffer(out)
    except Exception as e:
        raise ValueError(f"Error happened while transpose op: {e}")

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        # TODO: record needs_input_grad, requires_grad, and parents for backprop
        flags = [tensor.requires_grad for tensor in tensors]
        self.needs_input_grad = flags

        if any(flag is True for flag in flags):
            self.requires_grad = True
        elif None in flags:
            self.requires_grad = None
        else:
            self.requires_grad = False
        
        if self.requires_grad is True:
            self.parents = tensors

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")

    Function.forward = forward
    Function.backward = backward
    return Function

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    # TODO: build the Function, run forward on the input buffers, wrap in a
    # Tensor, and link out._ctx when a gradient is needed.
    ctx = cls(*tensors)
    buf = [t.lazydata for t in tensors]
    out_buf = ctx.forward(*buf, **kwargs)
    result = Tensor(out_buf, requires_grad=ctx.requires_grad)
    if ctx.requires_grad is True:
        result._ctx = ctx
    return result


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg (not yet solved)
# TODO: implement

# Step 17 - Relu (not yet solved)
# TODO: implement

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

