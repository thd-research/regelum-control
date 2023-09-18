"""Contains auxiliary tools."""

# TODO: THIS DESCRIPTION IS TOO SHORT. EXTEND IT

import inspect
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from enum import IntEnum
from numpy.random import rand
from scipy import signal
from typing import Union

try:
    import casadi

    CASADI_TYPES = tuple(
        x[1] for x in inspect.getmembers(casadi.casadi, inspect.isclass)
    )
except (ModuleNotFoundError, AttributeError):
    CASADI_TYPES = tuple()
import types

try:
    import torch

    TORCH_TYPES = tuple(
        x[1]
        for x in inspect.getmembers(torch, inspect.isclass)
        if ("torch" in str(x[1]))
    )
except ModuleNotFoundError:
    TORCH_TYPES = tuple()


# TODO: MISSING ENTRANCE SENTENCE: this class is bla-bla, it is needed to do bla-bla
class RCType(IntEnum):
    """Type inference proceeds by priority: `Torch` type has priority 3, `CasADi` type has priority 2, `NumPy` type has priority 1.

    That is, if, for instance, a function of two arguments gets an argument of a `NumPy` type and an argument of a `CasAdi` type,
    then the function's output type is inferred as a `CasADi` type.
    Mixture of CasADi types will raise a `TypeError` exception.
    """

    TORCH = 3
    CASADI = 2
    NUMPY = 1


TORCH = RCType.TORCH
CASADI = RCType.CASADI
NUMPY = RCType.NUMPY


def torch_safe_log(x, eps=1e-10):
    return torch.log(x + eps)


def is_CasADi_typecheck(*args) -> Union[RCType, bool]:
    return CASADI if any([isinstance(arg, CASADI_TYPES) for arg in args]) else False


def is_Torch_typecheck(*args) -> Union[RCType, bool]:
    return TORCH if any([isinstance(arg, TORCH_TYPES) for arg in args]) else False


# TODO: ADD DOCSTRING
def type_inference(*args, **kwargs) -> Union[RCType, bool]:
    is_CasADi = is_CasADi_typecheck(*args, *kwargs.values())
    is_Torch = is_Torch_typecheck(*args, *kwargs.values())
    if is_CasADi + is_Torch > 4:
        raise TypeError(
            "There is no support for simultaneous usage of both NumPy and CasADi"
        )
    else:
        result_type = max(is_CasADi, is_Torch, NUMPY)
        return result_type


# TODO: ADD DOCSTRING
def safe_unpack(argin):
    if isinstance(argin, (list, tuple)):
        return argin
    else:
        return (argin,)


# TODO: ADD DOCSTRING
def decorateAll(decorator):
    class MetaClassDecorator(type):
        def __new__(cls, classname, supers, classdict):
            for name, elem in classdict.items():
                if (
                    isinstance(elem, types.FunctionType)
                    and (name != "__init__")
                    and not isinstance(elem, staticmethod)
                ):
                    classdict[name] = decorator(classdict[name])
            return type.__new__(cls, classname, supers, classdict)

    return MetaClassDecorator


# TODO: ADD DOCSTRING
@decorateAll
def metaclassTypeInferenceDecorator(function):
    def wrapper(*args, **kwargs):
        rc_type = kwargs.get("rc_type")
        if rc_type is not None:
            del kwargs["rc_type"]
            return function(*args, **kwargs, rc_type=rc_type)
        else:
            return function(*args, **kwargs, rc_type=type_inference(*args, **kwargs))

    return wrapper


class Clock:
    def __init__(self, period: float, time_start: float = 0.0, eps=1e-7):
        self.period = period
        self.eps = eps
        self.time_start = time_start
        self.reset()

    def check_time(self, time: float):
        self.delta_time = time - self.current_time
        self.current_time = time

        if (
            self.is_first_time_called
            or self.current_time > self.last_sampled_time + self.period - self.eps
        ):
            self.last_sampled_time = time
            result = True
        else:
            result = False

        self.is_first_time_called = False
        return result

    def reset(self):
        self.last_sampled_time = self.current_time = self.time_start
        self.is_first_time_called = True
        self.delta_time = 0.0


# TODO: ADD DOCSTRING
class RCTypeHandler(metaclass=metaclassTypeInferenceDecorator):
    TORCH = RCType.TORCH
    CASADI = RCType.CASADI
    NUMPY = RCType.NUMPY

    def LeakyReLU(self, x, negative_slope=0.01, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.maximum(0, x) + negative_slope * np.minimum(0, x)
        elif rc_type == TORCH:
            return torch.nn.LeakyReLU(negative_slope=negative_slope)(x)
        elif rc_type == CASADI:
            return self.max(
                [self.zeros(self.shape(x), prototype=x), x]
            ) + negative_slope * self.min([self.zeros(self.shape(x), prototype=x), x])

    def CasADi_primitive(self, type: str = "MX", rc_type: RCType = NUMPY):
        if type == "MX":
            return casadi.MX.sym("x", 1)
        elif type == "SX":
            return casadi.SX.sym("x", 1)
        elif type == "DM":
            return casadi.DM([0])

    def cos(self, x, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.cos(x)
        elif rc_type == TORCH:
            return torch.cos(x)
        elif rc_type == CASADI:
            return casadi.cos(x)

    def clip(self, x, l, u, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.clip(x, l, u)
        elif rc_type == TORCH:
            return torch.clip(x, l, u)
        elif rc_type == CASADI:
            return casadi.fmin(
                self.concatenate([casadi.max(self.concatenate([x, l])), u])
            )

    def diag(self, x, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            diag = np.diag(x)
            if len(diag.shape) == 1:
                return diag.reshape(-1, 1)
            else:
                return diag
        elif rc_type == TORCH:
            diag = torch.diag(x)
            if len(diag.shape) == 1:
                return diag.reshape(-1, 1)
            else:
                return diag
        elif rc_type == CASADI:
            return casadi.diag(x)

    def sin(self, x, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.sin(x)
        elif rc_type == TORCH:
            return torch.sin(x)
        elif rc_type == CASADI:
            return casadi.sin(x)

    def floor(self, x, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.floor(x)
        elif rc_type == TORCH:
            return torch.floor(x)
        elif rc_type == CASADI:
            return casadi.floor(x)

    def column_stack(self, tup, rc_type: RCType = NUMPY):
        rc_type = type_inference(*tup)

        if rc_type == NUMPY:
            return np.column_stack(tup)
        elif rc_type == TORCH:
            return torch.column_stack(tup)
        elif rc_type == CASADI:
            return casadi.horzcat(*tup)

    def hstack(self, tup, rc_type: RCType = NUMPY):
        rc_type = type_inference(*tup)

        if rc_type == NUMPY:
            return np.hstack(tup)
        elif rc_type == TORCH:
            return torch.hstack(tup)
        elif rc_type == CASADI:
            return casadi.horzcat(*tup)

    def vstack(self, tup, rc_type: RCType = NUMPY):
        rc_type = type_inference(*tup)

        if rc_type == NUMPY:
            return np.vstack(tup)
        elif rc_type == TORCH:
            return torch.vstack(tup)
        elif rc_type == CASADI:
            return casadi.vertcat(*tup)

    def exp(self, x, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.exp(x)
        elif rc_type == TORCH:
            return torch.exp(x)
        elif rc_type == CASADI:
            return casadi.exp(x)

    def log(self, x, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.log(x)
        elif rc_type == TORCH:
            return torch.log(x)
        elif rc_type == CASADI:
            return casadi.log(x)

    def penalty_function(self, x, penalty_coeff=3, delta=1, rc_type: RCType = NUMPY):
        return self.exp(penalty_coeff * (x - delta))

    def push_vec(self, matrix, vec, rc_type: RCType = NUMPY):
        return self.column_stack([matrix[:, 1:], vec], rc_type=rc_type)

    def reshape_CasADi_as_np(self, array, dim_params, rc_type: RCType = NUMPY):
        result = self.zeros(dim_params, prototype=array)
        n_rows, n_cols = dim_params
        array_n_rows, array_n_cols = self.shape(array)

        for i in range(n_rows):
            for j in range(n_cols):
                result[i, j] = array[
                    (i * n_cols + j) // array_n_cols, (i * n_cols + j) % array_n_cols
                ]

        return result

    def reshape_to_column(self, array, length, rc_type: RCType = NUMPY):
        result_array = rc.reshape(array, [length, 1])
        return result_array

    def reshape(
        self, array, dim_params: Union[list, tuple, int], rc_type: RCType = NUMPY
    ):
        if rc_type == CASADI:
            if isinstance(dim_params, (list, tuple)):
                if len(dim_params) > 1:
                    return self.reshape_CasADi_as_np(array, dim_params)
                else:
                    return casadi.reshape(array, dim_params[0], 1)
            elif isinstance(dim_params, int):
                return casadi.reshape(array, dim_params, 1)
            else:
                raise TypeError(
                    "Wrong type of dimension parameter was passed.\
                         Possible cases are: int, [int], [int, int, ...]"
                )

        elif rc_type == NUMPY:
            return np.reshape(array, dim_params)

        elif rc_type == TORCH:
            return torch.reshape(array, dim_params)

    def array(
        self, array, prototype=None, rc_type: RCType = NUMPY, _force_numeric=False
    ):
        if isinstance(prototype, (list, tuple)):
            rc_type = type_inference(*prototype)

        if rc_type == NUMPY:
            self._array = np.array(array)
        elif rc_type == TORCH:
            if hasattr(prototype, "device"):
                device = prototype.device
            elif hasattr(array, "device"):
                device = array.device
            else:
                device = torch.device("cpu")

            self._array = torch.tensor(array, device=device)
        elif rc_type == CASADI:
            if _force_numeric:
                self._array = casadi.DM(array)
            else:
                casadi_constructor = (
                    type(prototype) if prototype is not None else casadi.DM
                )

                self._array = casadi_constructor(array)

        return self._array

    def ones(
        self,
        argin,
        prototype=None,
        rc_type: RCType = NUMPY,
    ):
        if isinstance(prototype, (list, tuple)):
            rc_type = type_inference(*prototype)

        if rc_type == NUMPY:
            self._array = np.ones(argin)
        elif rc_type == TORCH:
            self._array = torch.ones(argin)
        elif rc_type == CASADI:
            if isinstance(prototype, (list, tuple)):
                casadi_constructor = casadi.DM

                for constructor_type in map(lambda x: type(x), prototype):
                    if constructor_type == casadi.MX:
                        casadi_constructor = casadi.MX
                        break
            else:
                casadi_constructor = (
                    type(prototype) if prototype is not None else casadi.DM
                )

            self._array = casadi_constructor.ones(*safe_unpack(argin))

        return self._array

    def zeros(
        self,
        argin,
        prototype=None,
        rc_type: RCType = NUMPY,
    ):
        if isinstance(prototype, (list, tuple)):
            rc_type = type_inference(*prototype)

        if rc_type == NUMPY:
            return np.zeros(argin)
        elif rc_type == TORCH:
            return torch.zeros(argin)
        elif rc_type == CASADI:
            if isinstance(prototype, (list, tuple)):
                casadi_constructor = casadi.DM

                for constructor_type in map(lambda x: type(x), prototype):
                    if constructor_type == casadi.MX:
                        casadi_constructor = casadi.MX
                        break
            else:
                casadi_constructor = (
                    type(prototype) if prototype is not None else casadi.DM
                )

            self._array = casadi_constructor.zeros(*safe_unpack(argin))

            return self._array

    def concatenate(self, argin, rc_type: Union[RCType, bool] = NUMPY, axis=0):
        rc_type = type_inference(*safe_unpack(argin))
        if rc_type == NUMPY:
            return np.concatenate(argin, axis=axis)
        elif rc_type == TORCH:
            return torch.cat(argin, dim=axis)
        elif rc_type == CASADI:
            if isinstance(argin, (list, tuple)):
                if axis == 0:
                    return casadi.vertcat(*argin)
                elif axis == 1:
                    return casadi.horzcat(*argin)
                else:
                    raise ValueError("Not implemented value of axis for CasADi")

    def atleast_1d(self, dim, rc_type: RCType = NUMPY):
        return np.atleast_1d(dim)

    def transpose(self, A, rc_type: RCType = NUMPY):
        if rc_type == TORCH:
            return A.mT if len(A.shape) > 1 else A.T
        else:
            return A.T

    def vec(self, expr, rc_type: RCType = NUMPY):
        if rc_type == CASADI:
            return casadi.vec(expr)
        else:
            return expr

    def rep_mat(self, array, n, m, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.tile(array, (n, m))
        elif rc_type == TORCH:
            return torch.tile(array, (n, m))
        elif rc_type == CASADI:
            return casadi.repmat(array, n, m)

    def matmul(self, A, B, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.matmul(A, B)
        elif rc_type == TORCH:
            A = torch.tensor(A).double()
            B = torch.tensor(B).double()
            return torch.matmul(A, B)
        elif rc_type == CASADI:
            return casadi.mtimes(A, B)

    def casadi_outer(self, v1, v2, rc_type: RCType = NUMPY):
        if not is_CasADi_typecheck(v1):
            v1 = self.array_symb(v1)

        return casadi.horzcat(*[v1 * v2_i for v2_i in v2.nz])

    def outer(self, v1, v2, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.outer(v1, v2)
        elif rc_type == TORCH:
            return torch.outer(v1, v2)
        elif rc_type == CASADI:
            return self.casadi_outer(v1, v2)

    def sign(self, x, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.sign(x)
        elif rc_type == TORCH:
            return torch.sign(x)
        elif rc_type == CASADI:
            return casadi.sign(x)

    def abs(self, x, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.abs(x)
        elif rc_type == TORCH:
            return torch.abs(x)
        elif rc_type == CASADI:
            return casadi.fabs(x)

    def min(self, array, rc_type: RCType = NUMPY):
        if isinstance(array, (list, tuple)):
            rc_type = type_inference(*array)

        if rc_type == NUMPY:
            return np.min(array)
        elif rc_type == TORCH:
            return torch.min(array)
        elif rc_type == CASADI:
            return casadi.fmin(*safe_unpack(array))

    def max(self, array, rc_type: RCType = NUMPY):
        if isinstance(array, (list, tuple)):
            rc_type = type_inference(*array)

        if rc_type == NUMPY:
            return np.max(array)
        elif rc_type == TORCH:
            return torch.max(array)
        elif rc_type == CASADI:
            return casadi.mmax(*safe_unpack(array))

    def sum_2(self, array, rc_type: RCType = NUMPY):
        if isinstance(array, (list, tuple)):
            rc_type = type_inference(*array)

        if rc_type == NUMPY:
            return np.sum(array**2)
        elif rc_type == TORCH:
            return torch.sum(array**2)
        elif rc_type == CASADI:
            return casadi.sum1(array**2)

    def sum(self, array, rc_type: RCType = NUMPY, axis=None):
        if isinstance(array, (list, tuple)):
            rc_type = type_inference(*array)

        if rc_type == NUMPY:
            return np.sum(array, axis=axis)
        elif rc_type == TORCH:
            return torch.sum(array, dim=axis)
        elif rc_type == CASADI:
            if axis is None:
                return casadi.sum2(casadi.sum1(array))
            if axis == 0:
                return casadi.sum1(array)
            if axis == 1:
                return casadi.sum2(array)

    def mean(self, array, rc_type: RCType = NUMPY):
        if isinstance(array, (list, tuple)):
            rc_type = type_inference(*array)

        if rc_type == NUMPY:
            return np.mean(array)
        elif rc_type == TORCH:
            return torch.mean(array)
        elif rc_type == CASADI:
            length = self.max(self.shape(*safe_unpack(array)))
            return casadi.sum1(*safe_unpack(array)) / length

    def force_column(self, argin, rc_type: RCType = NUMPY):
        assert len(argin.shape) <= 2, "Only 1D and 2D arrays are supported."

        if rc_type == CASADI:
            if argin.shape[1] > argin.shape[0] and argin.shape[0] == 1:
                return argin.T
            else:
                return argin
        else:
            return argin.reshape(-1, 1)

    def force_row(self, argin, rc_type: RCType = NUMPY):
        assert len(argin.shape) <= 2, "Only 1D and 2D arrays are supported."

        if rc_type == CASADI:
            if argin.shape[0] > argin.shape[1] and argin.shape[1] == 1:
                return argin.T
            else:
                return argin
        else:
            return argin.reshape(1, -1)

    def cross(self, A, B, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.cross(A, B)
        elif rc_type == TORCH:
            return torch.cross(A, B)
        elif rc_type == CASADI:
            return casadi.cross(A, B)

    def dot(self, A, B, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.dot(A, B)
        elif rc_type == TORCH:
            return torch.dot(A, B)
        elif rc_type == CASADI:
            return casadi.dot(A, B)

    def sqrt(self, x, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.sqrt(x)
        elif rc_type == TORCH:
            return torch.sqrt(x)
        elif rc_type == CASADI:
            return casadi.sqrt(x)

    def shape(self, array, rc_type: RCType = NUMPY):
        if rc_type == CASADI:
            return array.size()
        elif rc_type == NUMPY:
            return np.shape(array)
        elif rc_type == TORCH:
            return array.size()

    def function_to_lambda_with_params(
        self, function_to_lambda, *params, var_prototype=None, rc_type: RCType = NUMPY
    ):
        if rc_type in (NUMPY, TORCH):
            if params:
                return lambda x: function_to_lambda(x, *params)
            else:
                return lambda x: function_to_lambda(x)
        else:
            try:
                x_symb = self.array_symb(self.shape(var_prototype))
            except NotImplementedError:
                x_symb = self.array_symb((*safe_unpack(self.shape(var_prototype)), 1))

            if params:
                return function_to_lambda(x_symb, *safe_unpack(params)), x_symb
            else:
                return function_to_lambda(x_symb), x_symb

    def lambda2symb(self, lambda_function, *x_symb, rc_type: RCType = NUMPY):
        return lambda_function(*x_symb)

    def torch_tensor(self, x, requires_grad=True, rc_type: RCType = NUMPY):
        return torch.tensor(x, requires_grad=requires_grad)

    def add_torch_grad(x, rc_type: RCType = NUMPY):
        if rc_type == TORCH:
            x.requires_grad = True
        else:
            raise TypeError("Cannot assign grad to non-torch type variable")

    def tanh(self, x, rc_type: RCType = NUMPY):
        if rc_type == CASADI:
            res = casadi.tanh(x)
        elif rc_type == NUMPY:
            res = np.tanh(x)
        elif rc_type == TORCH:
            res = torch.tanh(x)
        return res

    def if_else(self, c, x, y, rc_type: RCType = NUMPY):
        if rc_type == CASADI:
            res = casadi.if_else(c, x, y)
            return res
        elif rc_type == TORCH or rc_type == NUMPY:
            return x if c else y

    def kron(self, A, B, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.kron(A, B)
        elif rc_type == TORCH:
            return torch.kron(A, B)
        elif rc_type == CASADI:
            return casadi.kron(A, B)

    def array_symb(
        self, tup=None, literal="x", rc_type: RCType = NUMPY, prototype=None
    ):
        if prototype is not None:
            shape = self.shape(prototype)
        else:
            shape = tup

        if isinstance(shape, tuple):
            if len(tup) > 2:
                raise ValueError(
                    f"Not implemented for number of dimensions greater than 2. Passed: {len(tup)}"
                )
            else:
                return casadi.MX.sym(literal, *tup)

        elif isinstance(tup, int):
            return casadi.MX.sym(literal, tup)

        else:
            raise TypeError(
                f"Passed an invalide argument of type {type(tup)}. Takes either int or tuple data types"
            )

    def norm_1(self, v, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.linalg.norm(v, 1)
        elif rc_type == TORCH:
            return torch.linalg.norm(v, 1)
        elif rc_type == CASADI:
            return casadi.norm_1(v)

    def norm_2(self, v, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.linalg.norm(v, 2)
        elif rc_type == TORCH:
            return torch.linalg.norm(v, 2)
        elif rc_type == CASADI:
            return casadi.norm_2(v)

    def logic_and(self, a, b, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.logical_and(a, b)
        elif rc_type == TORCH:
            return torch.logical_and(a, b)
        elif rc_type == CASADI:
            return casadi.logic_and(a, b)

    def to_np_1D(self, v, rc_type: RCType = NUMPY):
        if rc_type == CASADI:
            return v.T.full().flatten()
        else:
            return v

    def squeeze(self, v, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.squeeze(v)
        elif rc_type == TORCH:
            return torch.squeeze(v)
        elif rc_type == CASADI:
            assert (
                v.shape[0] == 1 or v.shape[1] == 1
            ), "Only columns and rows are supported."
            if v.shape[0] == 1 and v.shape[1] > 1:
                return v.T
            else:
                return v

    def uptria2vec(self, mat, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            result = mat[np.triu_indices(self.shape(mat)[0])]
            return result
        elif rc_type == TORCH:
            result = mat[torch.triu_indices(self.shape(mat)[0])]
            return result
        elif rc_type == CASADI:
            n = self.shape(mat)[0]

            vec = rc.zeros((int(n * (n + 1) / 2)), prototype=mat)

            k = 0
            for i in range(n):
                for j in range(i, n):
                    vec[k] = mat[i, j]
                    k += 1

            return vec

    def append(self, array, to_append, rc_type: RCType = NUMPY):
        if rc_type == NUMPY:
            return np.append(array, to_append)

    # TODO: DO WE REALLY NEED THIS DM? WHY NOT TO USE HUMAN READABLE TERMINOLOGY? SAY CASADI_NUMERIC, CASADI_SYMB, CASADI_MATSYMB
    @staticmethod
    def DM(mat):
        return casadi.DM(mat)

    @staticmethod
    def SX(mat):
        return casadi.SX(mat)

    @staticmethod
    def MX(mat):
        return casadi.MX(mat)

    def autograd(self, function, x, *args, rc_type: RCType = NUMPY):
        return casadi.Function(
            "f", [x, *args], [casadi.gradient(function(x, *args), x)]
        )

    def to_casadi_function(
        self, symbolic_expression, symbolic_var, rc_type: RCType = NUMPY
    ):
        return casadi.Function("f", [symbolic_var], [symbolic_expression])

    def soft_abs(self, x, a=20, rc_type: RCType = NUMPY):
        return a * rc.abs(x) ** 3 / (1 + a * x**2)


rc = RCTypeHandler()


# TODO: ADD DOCSTRING??
def simulation_progress(bar_length=10, print_level=100):
    counter = 0

    def simulation_progress_inner(step_function):
        def wrapper(self, *args, **kwargs):
            nonlocal counter

            result = step_function(self, *args, **kwargs)

            current_time = self.time
            if counter % print_level == 0:
                bar = ["." for _ in range(bar_length)]
                final_time = self.time_final
                part_done = int(current_time / final_time * bar_length)
                bar = ["#" for i in range(part_done)] + bar[part_done:]
                print(
                    "".join(bar),
                    f"Episode is {int(current_time / final_time*100)}% done.\nSimulation time {current_time:.2f}",
                )

            counter += 1

            if result == -1:
                counter = 0
                print("End of episode")
            return result

        return wrapper

    return simulation_progress_inner


# class CASADI_vector_convention:
#     """
#     A context manager for automatic transpose of vectors for vector_convention with the CASADI default vector dimension convention.
#     regelum treats vectors, just like numpy, as rows.
#     CASADI, on contrary, treats them as columns.
#     We want to flatten everything to the desfault assumed standard which is row.
#     But in order to process some code blocks so that they comply with CASADI, we use this context manager.

#     Basic usage goes like:

#     ..  code-block:: python

#         with CASADI_vector_convention(locals()):
#             Dstate = rc.zeros(dim_state, rc_type: RCType = rc.CASADI)
#             ...

#     This code will automatically set `Dstate` to the column format, whenever the backround engine is CASADI.
#     After the body of the context is executed, `Dstate` will be converted to the default (row) format.
#     """

#     def __init__(self, local_dict):
#         self.local_dict = local_dict
#         RCTypeHandler.is_force_row = False

#     def __enter__(self):
#         self.local_dict["rc"] = rc

#     def __exit__(self, *args):
#         print(self.local_dict)
#         for var in self.local_dict:
#             if is_CasADi_typecheck(self.local_dict[var]):
#                 try:
#                     shape = rc.shape(self.local_dict[var])
#                     if shape[1] == 1:
#                         self.local_dict[var] = rc.force_row(self.local_dict[var])
#                         break
#                 except:
#                     pass
#             else:
#                 pass

#         RCTypeHandler.is_force_row = True


# TODO: REMOVE THESE?
def rej_sampling_rvs(dim, pdf, M):
    r"""Random variable (pseudo)-realizations via rejection sampling.

    Parameters
    ----------
    dim : : integer
        dimension of the random variable
    pdf : : function
        desired probability density function
    M : : number greater than 1
        it must hold that :math:`\text{pdf}_{\text{desired}} \le M \text{pdf}_{\text{proposal}}`.
        This function uses a normal pdf with zero mean and identity covariance matrix as a proposal distribution.
        The smaller `M` is, the fewer iterations to produce a sample are expected.

    Returns
    -------
    A single realization (in general, as a vector) of the random variable with the desired probability density.

    """
    # Use normal pdf with zero mean and identity covariance matrix as a proposal distribution
    normal_RV = st.multivariate_normal(cov=np.eye(dim))

    # Bound the number of iterations to avoid too long loops
    max_iters = 1e3

    curr_iter = 0

    while curr_iter <= max_iters:
        proposal_sample = normal_RV.rvs()

        unif_sample = rand()

        if unif_sample < pdf(proposal_sample) / M / normal_RV.pdf(proposal_sample):
            return proposal_sample


def push_vec(matrix, vec):
    return rc.vstack([matrix[1:, :], vec.T])


class ZOH:
    """Zero-order hold."""

    def __init__(self, init_time=0, init_val=0, sample_time=1):
        self.time_step = init_time
        self.sample_time = sample_time
        self.currVal = init_val

    def hold(self, signal_val, time):
        timeInSample = time - self.time_step
        if timeInSample >= self.sample_time:  # New sample
            self.time_step = time
            self.currVal = signal_val

        return self.currVal


class DFilter:
    """Real-time digital filter."""

    def __init__(
        self,
        filter_num,
        filter_den,
        data_buffer_size=16,
        init_time=0,
        init_val=0,
        sample_time=1,
    ):
        self.Num = filter_num
        self.Den = filter_den
        self.zi = rc.rep_mat(
            signal.lfilter_zi(filter_num, filter_den), 1, init_val.size
        )

        self.time_step = init_time
        self.sample_time = sample_time
        self.buffer = rc.rep_mat(init_val, 1, data_buffer_size)

    def filt(self, signal_val, time=None):
        # Sample only if time is specified
        if time is not None:
            timeInSample = time - self.time_step
            if timeInSample >= self.sample_time:  # New sample
                self.time_step = time
                self.buffer = push_vec(self.buffer, signal_val)
        else:
            self.buffer = push_vec(self.buffer, signal_val)

        bufferFiltered = np.zeros(self.buffer.shape)

        for k in range(0, signal_val.size):
            bufferFiltered[k, :], self.zi[k] = signal.lfilter(
                self.Num, self.Den, self.buffer[k, :], zi=self.zi[k, :]
            )
        return bufferFiltered[-1, :]


def dss_sim(A, B, C, D, uSqn, initial_guess, y0):
    """Simulate output response of a discrete-time state-space model."""
    if uSqn.ndim == 1:
        return y0, initial_guess
    else:
        ySqn = np.zeros([uSqn.shape[0], C.shape[0]])
        xSqn = np.zeros([uSqn.shape[0], A.shape[0]])
        x = initial_guess
        ySqn[0, :] = y0
        xSqn[0, :] = initial_guess
        for k in range(1, uSqn.shape[0]):
            x = A @ x + B @ uSqn[k - 1, :]
            xSqn[k, :] = x
            ySqn[k, :] = C @ x + D @ uSqn[k - 1, :]

        return ySqn, xSqn


# TODO: CHECK IF THESE ARE NEEDED
def update_line(line, newX, newY):
    line.set_xdata(np.append(line.get_xdata(), newX))
    line.set_ydata(np.append(line.get_ydata(), newY))


def reset_line(line):
    line.set_data([], [])


def update_scatter(scatter, newX, newY):
    scatter.set_offsets(np.vstack([scatter.get_offsets().data, np.c_[newX, newY]]))


def update_text(textHandle, newText):
    textHandle.set_text(newText)


def update_patch(patchHandle, new_color):
    patchHandle.set_color(str(new_color))


def on_key_press(event, anm):
    """Key press event handler for a ``FuncAnimation`` animation object."""
    if event.key == " ":
        if anm.running:
            anm.event_source.stop()

        else:
            anm.event_source.start()
        anm.running ^= True
    elif event.key == "q" or event.key:
        if anm is not None:
            anm.event_source.stop()
        plt.clf()
        plt.cla()
        plt.close()

        raise Exception("Script terminated after q key press")


def on_close(event):
    raise Exception("Script terminated after animation was closed")


log = None