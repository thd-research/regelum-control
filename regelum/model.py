"""Contains models.

These can be used in system dynamics fitting, critic and other tasks.

Updates to come.

"""
from copy import deepcopy

import regelum


from .__utilities import rc

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.distributions.multivariate_normal import MultivariateNormal
except ModuleNotFoundError:
    from unittest.mock import MagicMock

    torch = MagicMock()
    nn = MagicMock()
    F = MagicMock()
    MultivariateNormal = MagicMock()

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, Tuple
import numpy as np
import casadi as cs


def force_positive_def(func):
    def positive_def_wrapper(self, *args, **kwargs):
        if self.force_positive_def:
            return rc.soft_abs(func(self, *args, **kwargs))
        else:
            return func(self, *args, **kwargs)

    return positive_def_wrapper


class Model(regelum.RegelumBase, ABC):
    """Blueprint of a model."""

    def __call__(self, *argin, weights=None, use_stored_weights=False):
        if len(argin) == 2:
            left, right = argin
            if len(left.shape) != len(right.shape):
                raise ValueError(
                    "In Model.__call__ left and right arguments must have same number of dimensions!"
                )

            dim = len(left.shape)

            if dim == 1:
                argin = rc.concatenate(argin, axis=0)
            elif dim == 2:
                argin = rc.concatenate(argin, axis=1)
            else:
                raise ValueError("Wrong number of dimensions in Model.__call__")
        elif len(argin) == 1:
            argin = argin[0]
        else:
            raise ValueError(
                f"Wrong number of arguments in Model.__call__. Can be either 1 or 2. Got: {len(argin)}"
            )

        if use_stored_weights is False:
            if weights is not None:
                result = self.forward(argin, weights=weights)
            else:
                result = self.forward(argin)
        else:
            result = self.cache.forward(argin)

        return result

    @property
    def weights(self):
        return self._weights

    @property
    def named_parameters(self):
        return self._weights

    @property
    @abstractmethod
    def model_name(self):
        return "model_name"

    @abstractmethod
    def __init__(self):
        """Initialize an instance of a model."""
        pass

    @abstractmethod
    def forward(self):
        pass

    def update_weights(self, weights):
        self._weights = weights

    def cache_weights(self, weights=None):
        if "cache" not in self.__dict__.keys():
            self.cache = deepcopy(self)

        if weights is None:
            self.cache.update_weights(self._weights)
        else:
            self.cache.update_weights(weights)

    def update_and_cache_weights(self, weights):
        self.cache_weights(weights)
        self.update_weights(weights)

    def restore_weights(self):
        """Assign the weights of the cached model to the active model.

        This may be needed when pytorch optimizer resulted in unsatisfactory weights, for instance.

        """
        self.update_and_cache_weights(self.cache.weights)


class ModelQuadLin(Model):
    """Base class for generic quadratic linear models.

    Normally used for running objective specification (diagonal quadratic matrix without linear terms) and critic polynomial models.
    """

    model_name = "ModelQuadLin"

    def __init__(
        self,
        quad_matrix_type: str,
        is_with_linear_terms: bool = False,
        dim_inputs: int = None,
        weights: np.array = None,
        weight_min: float = 1.0e-6,
        weight_max: float = 1.0e3,
    ):
        """Initialize an instance of quadratic-linear model.

        :param quad_matrix_type: Type of quadratic matrix. Can be 'diagonal', 'full' or 'symmetric'.
        :type quad_matrix_type: str
        :param is_with_linear_terms: Whether include linear terms or not, defaults to False
        :type is_with_linear_terms: bool, optional
        :param dim_inputs: Dimension of system's (agent's) inputs, defaults to None
        :type dim_inputs: int, optional
        :param weights: Manual set of model weights, defaults to None
        :type weights: _type_, optional
        :param weight_min: Lower bound for weights, defaults to 1.0e-6
        :type weight_min: float, optional
        :param weight_max: Upper bound for weights, defaults to 1.0e3
        :type weight_max: float, optional
        """
        assert (
            dim_inputs is not None or weights is not None
        ), "Need dim_inputs or weights"

        self.quad_matrix_type = quad_matrix_type
        self.is_with_linear_terms = is_with_linear_terms
        self.single_weight_min = weight_min
        self.single_weight_max = weight_max

        if weights is None:
            self._calculate_dims(dim_inputs)
            self.weight_min = weight_min * rc.ones(self.dim_weights)
            self.weight_max = weight_max * rc.ones(self.dim_weights)
            self._weights = (self.weight_min + self.weight_max) / 20.0
        else:
            self._calculate_dims(self._calculate_dim_inputs(len(weights)))
            assert self.dim_weights == len(weights), "Wrong shape of dim_weights"
            self._weights = rc.array(weights)

        self.update_and_cache_weights(self._weights)

    @property
    def weight_bounds(self):
        return rc.array([[self.single_weight_min, self.single_weight_max]])

    def _calculate_dim_inputs(self, dim_weights):
        if self.quad_matrix_type == "diagonal":
            if self.is_with_linear_terms:
                return dim_weights // 2
            else:
                return dim_weights
        elif self.quad_matrix_type == "full":
            if self.is_with_linear_terms:
                return round((np.sqrt(1 + 4 * dim_weights) - 1) / 2)
            else:
                return round(np.sqrt(dim_weights))
        elif self.quad_matrix_type == "symmetric":
            if self.is_with_linear_terms:
                return round((np.sqrt(9 + 8 * dim_weights) - 1) / 2)
            else:
                return round((np.sqrt(1 + 8 * dim_weights) - 1) / 2)

    def _calculate_dims(self, dim_inputs):
        self.dim_inputs = dim_inputs
        self.dim_linear = dim_inputs if self.is_with_linear_terms else 0
        if self.quad_matrix_type == "diagonal":
            self.dim_quad = dim_inputs
        elif self.quad_matrix_type == "full":
            self.dim_quad = dim_inputs * dim_inputs
        elif self.quad_matrix_type == "symmetric":
            self.dim_quad = dim_inputs * (dim_inputs + 1) // 2

        self.dim_weights = self.dim_quad + self.dim_linear

    def cast_to_inputs_type(self, value, inputs):
        if value is None:
            return None
        if isinstance(inputs, torch.Tensor):
            device = inputs.device
            if not isinstance(value, torch.Tensor):
                return torch.FloatTensor(value).to(device)
            elif device != value.device:
                return value.to(device)
        if isinstance(inputs, cs.MX) and isinstance(value, np.ndarray):
            value = rc.DM(value)
        return value

    def forward_symmetric(self, inputs, weights):
        quad_matrix = ModelQuadLin.quad_matrix_from_flat_weights(
            weights[: self.dim_quad]
        )
        linear_coefs = (
            weights[None, self.dim_quad :] if self.is_with_linear_terms else None
        )

        return ModelQuadLin.quadratic_linear_form(
            inputs,
            self.cast_to_inputs_type(quad_matrix, inputs),
            self.cast_to_inputs_type(linear_coefs, inputs),
        )

    def forward_diagonal(self, inputs, weights):
        quad_matrix = rc.diag(weights[: self.dim_quad])
        linear_coefs = (
            weights[None, self.dim_quad :] if self.is_with_linear_terms else None
        )

        return ModelQuadLin.quadratic_linear_form(
            inputs,
            self.cast_to_inputs_type(quad_matrix, inputs),
            self.cast_to_inputs_type(linear_coefs, inputs),
        )

    def forward_full(self, inputs, weights):
        quad_matrix = rc.reshape(
            weights[: self.dim_quad], (self.dim_inputs, self.dim_inputs)
        )
        linear_coefs = (
            weights[None, self.dim_quad :] if self.is_with_linear_terms else None
        )

        return ModelQuadLin.quadratic_linear_form(
            inputs,
            self.cast_to_inputs_type(quad_matrix, inputs),
            self.cast_to_inputs_type(linear_coefs, inputs),
        )

    def forward(self, inputs, weights=None):
        if weights is None:
            weights = self._weights
        if self.quad_matrix_type == "symmetric":
            return self.forward_symmetric(inputs, weights)
        elif self.quad_matrix_type == "diagonal":
            return self.forward_diagonal(inputs, weights)
        elif self.quad_matrix_type == "full":
            return self.forward_full(inputs, weights)

    @staticmethod
    def quad_matrix_from_flat_weights(
        flat_weights: Union[np.array, cs.DM, torch.Tensor], tol=1e-7
    ):
        len_flat_weights = flat_weights.shape[0]
        dim_quad_matrix_float = (np.sqrt(1 + 8 * len_flat_weights) - 1) / 2
        dim_quad_matrix = round(dim_quad_matrix_float)
        assert np.isclose(
            dim_quad_matrix_float, dim_quad_matrix, tol
        ), f"Can't build quad matrix with flat_weights of dim {len_flat_weights}"

        quad_matrix = rc.zeros(
            (dim_quad_matrix, dim_quad_matrix), prototype=flat_weights
        )
        left_ids, right_ids = np.triu_indices(dim_quad_matrix)
        for weigth_idx, (i, j) in enumerate(zip(left_ids, right_ids)):
            quad_matrix[i, j] = flat_weights[weigth_idx]

        return quad_matrix

    @staticmethod
    def quadratic_linear_form(inputs, quad_matrix, linear_coefs=None):
        initial_dim_inputs = len(inputs.shape)
        assert (
            initial_dim_inputs == 1 or initial_dim_inputs == 2
        ), "Wrong shape of inputs can be 1d or 2d. Got {}".format(initial_dim_inputs)

        if initial_dim_inputs == 1:
            inputs = inputs.reshape(1, -1)
        assert (
            len(quad_matrix.shape) == 2
        ), "Wrong shape of quad matrix. Should be 2d. Got{}".format(
            len(quad_matrix.shape)
        )
        assert (
            quad_matrix.shape[0] == quad_matrix.shape[1]
        ), "Quad matrix should be square"
        assert (
            quad_matrix.shape[0] == inputs.shape[1]
        ), "Quad matrix should have same number of rows as inputs"

        quadratic_term = inputs @ quad_matrix @ inputs.T
        if len(quadratic_term.shape) > 0:
            quadratic_term = rc.diag(quadratic_term)

        if linear_coefs is not None:
            assert (
                len(linear_coefs.shape) == 2 and linear_coefs.shape[0] == 1
            ), "Wrong shape of linear coefs. Should be (1,n). Got {}".format(
                linear_coefs.shape
            )

            assert (
                quad_matrix.shape[1] == linear_coefs.shape[1]
            ), "Quad matrix should have same number of columns as linear coefs"

            linear_term = inputs @ linear_coefs.T
            output = quadratic_term + linear_term
        else:
            output = quadratic_term

        if initial_dim_inputs == 1:
            output = output.reshape(-1)
        return output


class ModelWeightContainer(Model):
    """Trivial model, which is typically used in actor in which actions are being optimized directly."""

    model_name = "action-sequence"

    def __init__(self, dim_output, weights_init=None):
        """Initialize an instance of a model returns weights on call independent of input.

        :param dim_input: input dimension
        :param single_weight_min: lower bound for every weight
        :param single_weight_max: upper bound for every weight
        """
        self.dim_output = dim_output
        self._weights = weights_init
        self.weights_init = weights_init
        self.update_and_cache_weights(self._weights)

    def forward(self, *argin, weights=None):
        if weights is not None:
            return rc.force_row(weights[0, : self.dim_output])
        else:
            return rc.force_row(self._weights[0, : self.dim_output])


class ModelNN(nn.Module):
    """Class of pytorch neural network models. This class is not to be used barebones.

    Instead, you should inherit from it and specify your concrete architecture.

    """

    model_name = "NN"

    @property
    def cache(self):
        """Isolate parameters of cached model from the current model."""
        return self.cached_model[0]

    def detach_weights(self):
        """Excludes the model's weights from the pytorch computation graph.

        This is needed to exclude the weights from the decision variables in optimization problems.
        An example is temporal-difference optimization, where the old critic is to be treated as a frozen model.

        """
        for variable in self.parameters():
            variable.detach_()

    def cache_weights(self, whatever=None):
        """Assign the active model weights to the cached model followed by a detach.

        This method also backs up itself and performs this operation only once upon the initialization procedure
        """
        if "cached_model" not in self.__dict__.keys():
            self.cached_model = (
                deepcopy(self),
            )  ## this is needed to prevent cached_model's parameters to be parsed by model init hooks

        self.cache.load_state_dict(self.state_dict())
        self.cache.detach_weights()

    @property
    def weights(self):
        return self.state_dict()

    def update_weights(self, whatever=None):
        pass

    def update_and_cache_weights(self, weights=None):
        if weights is not None:
            for item in weights:
                weights[item].requires_grad_()
            weights = self.load_state_dict(weights)
        self.cache_weights()

    def restore_weights(self):
        """Assign the weights of the cached model to the active model.

        This may be needed when pytorch optimizer resulted in unsatisfactory weights, for instance.

        """
        self.update_and_cache_weights(self.cache.state_dict())

    def __call__(self, *argin, weights=None, use_stored_weights=False):
        if len(argin) == 2:
            left, right = argin
            if len(left.shape) != len(right.shape):
                raise ValueError(
                    "In ModelNN.__call__ left and right arguments must have same number of dimensions!"
                )

            dim = len(left.shape)

            if dim == 1:
                argin = rc.concatenate(argin, axis=0)
            elif dim == 2:
                argin = rc.concatenate(argin, axis=1)
            else:
                raise ValueError("Wrong number of dimensions in ModelNN.__call__")
        elif len(argin) == 1:
            argin = argin[0]
        else:
            raise ValueError(
                f"Wrong number of arguments in ModelNN.__call__. Can be either 1 or 2. Got: {len(argin)}"
            )

        if use_stored_weights is False:
            if weights is not None:
                result = self.forward(argin, weights)
            else:
                result = self.forward(argin)
        else:
            result = self.cache.forward(argin)

        return result


class WeightClipper:
    """Weight clipper for pytorch layers."""

    def __init__(self, weight_min=None, weight_max=None):
        """Initialize a weight clipper.

        :param weight_min: minimum value for weight
        :param weight_max: maximum value for weight
        """
        self.weight_min = weight_min
        self.weight_max = weight_max

    def __call__(self, module):
        if self.weight_min is not None or self.weight_max is not None:
            # filter the variables to get the ones you want
            w = module.weight.data
            w = w.clamp(self.weight_min, self.weight_max)
            module.weight.data = w


class ModelPerceptron(ModelNN):
    """Helper class to ease the creation of perceptron models."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        leaky_relu_coef: float = 0.15,
        force_positive_def: bool = False,
        is_force_infinitesimal: bool = False,
        is_bias: bool = True,
        weight_max: Optional[float] = None,
        weight_min: Optional[float] = None,
        weights=None,
    ):
        """Initialize an instance of a fully-connected model.

        :param dim_input: dimension of model's input
        :type dim_input: int
        :param dim_output: dimension of model's output
        :type dim_output: int
        :param dim_hidden: number of neurons in hidden layers
        :type dim_hidden: int
        :param n_hidden_layers: number of hidden layers
        :type n_hidden_layers: int
        :param leaky_relu_coef: Leaky ReLU coefficient, defaults to 0.15
        :type leaky_relu_coef: float, optional
        :param force_positive_def: Whether to force positive definiteness of model through soft abs function, defaults to False
        :type force_positive_def: bool, optional
        :param is_force_infinitesimal: Make model equal to zero in the origin, defaults to False
        :type is_force_infinitesimal: bool, optional
        :param is_bias: Add bias, defaults to True
        :type is_bias: bool, optional
        :param weight_max: Upper bound for weights, defaults to None
        :type weight_max: Optional[float], optional
        :param weight_min: Lower bound for weights, defaults to None
        :type weight_min: Optional[float], optional
        :param weights: Pass weights manually with state_dict of torch module, defaults to None
        :type weights: _type_, optional
        """
        ModelNN.__init__(self)
        self.weight_clipper = WeightClipper(weight_min, weight_max)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_hidden_layers = n_hidden_layers
        self.leaky_relu_coef = leaky_relu_coef
        self.force_positive_def = force_positive_def
        self.is_force_infinitesimal = is_force_infinitesimal
        self.is_bias = is_bias
        self.input_layer = nn.Linear(dim_input, dim_hidden, bias=is_bias)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(dim_hidden, dim_hidden, bias=is_bias)
                for _ in range(n_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(dim_hidden, dim_output, bias=is_bias)

        if weights is not None:
            self.load_state_dict(weights)

        self.cache_weights()

    def _forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        self.input_layer.apply(self.weight_clipper)
        x = nn.functional.leaky_relu(
            self.input_layer(x), negative_slope=self.leaky_relu_coef
        )
        for hidden_layer in self.hidden_layers:
            hidden_layer.apply(self.weight_clipper)
            x = nn.functional.leaky_relu(
                hidden_layer(x), negative_slope=self.leaky_relu_coef
            )
        x = self.output_layer(x)

        return x

    @force_positive_def
    def forward(
        self, input_tensor: torch.FloatTensor, weights=None
    ) -> torch.FloatTensor:
        if self.is_force_infinitesimal:
            return self._forward(input_tensor) - self._forward(
                torch.zeros_like(input_tensor)
            )

        return self._forward(input_tensor)


class ModelWeightContainerTorch(ModelNN):
    """Pytorch model with forward that returns weights."""

    def __init__(
        self,
        dim_weights: Union[int, Tuple[int, int]],
        output_bounds: Optional[List[Any]] = None,
        output_bounding_type: str = "clip",
    ):
        """Instantiate ModelWeightContainerTorch.

        :param dim_weights: Dimensionality of the weights
        :type dim_weights: int
        :param output_bounds: Bounds of the output. If `None` output is not bounded, defaults to None
        :type output_bounds: Optional[List[Any]], optional
        """
        assert (
            output_bounding_type == "clip" or output_bounding_type == "tanh"
        ), "output_bounding_type must be 'clip' or 'tanh'"

        ModelNN.__init__(self)
        self.bounds_handler = (
            BoundsHandler(output_bounds) if output_bounds is not None else None
        )
        self.output_bounding_type = output_bounding_type

        self.dim_weights = (
            (1, dim_weights) if isinstance(dim_weights, int) else dim_weights
        )
        self._weights = torch.nn.Parameter(
            torch.FloatTensor(torch.zeros(self.dim_weights)),
            requires_grad=True,
        )

    def forward(self, inputs, weights=None):
        if self.bounds_handler is not None:
            if self.output_bounding_type == "clip":
                with torch.no_grad():
                    self._weights.clip_(-1.0, 1.0)

        if len(inputs.shape) == 1:
            inputs_like = self._weights[0, :]
        elif len(inputs.shape) == 2:
            if inputs.shape[0] <= self.dim_weights[0]:
                inputs_like = self._weights[: inputs.shape[0], :]
            else:
                raise ValueError(
                    f"ModelWeightContainerTorch: Wrong inputs shape! inputs.shape[0] (Got: {inputs.shape[0]}) should be <= dim_weights[0] (Got: {self.dim_weights[0]})."
                )
        else:
            raise ValueError("Wrong inputs shape! Can be either 1 or 2")

        if self.bounds_handler is not None:
            if self.output_bounding_type == "clip":
                # inputs_like are already clipped in the beggining of the function via WeightClipper
                return self.bounds_handler.unscale_from_minus_one_one_to_bounds(
                    inputs_like
                )
            elif self.output_bounding_type == "tanh":
                return self.bounds_handler.unscale_from_minus_one_one_to_bounds(
                    torch.tanh(inputs_like)
                )
        else:
            return inputs_like


class BoundsHandler(ModelNN):
    r"""Output layer for bounding the model's output. The formula is: math: `F^{-1}(\\tanh(x))`, where F is the linear transformation from `bounds` to [-1, 1]."""

    def __init__(self, bounds: Union[List[Any], np.array]):
        """Initialize an instance of NNOutputBounder.

        :param bounds: Bounds for the output.
        :type bounds: Union[List[Any], np.array]
        """
        ModelNN.__init__(self)
        self.register_parameter(
            name="__bounds",
            param=torch.nn.Parameter(
                torch.FloatTensor(bounds),
                requires_grad=False,
            ),
        )

    def get_unscale_coefs_from_minus_one_one_to_bounds(
        self,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        bounds = self.get_parameter("__bounds")

        unscale_bias, unscale_multiplier = (
            bounds.mean(dim=1),
            (bounds[:, 1] - bounds[:, 0]) / 2.0,
        )
        return unscale_bias, unscale_multiplier

    def unscale_from_minus_one_one_to_bounds(
        self, x: torch.FloatTensor
    ) -> torch.FloatTensor:
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_bounds()

        return x * unscale_multiplier + unscale_bias

    def scale_from_bounds_to_minus_one_one(self, y):
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_bounds()

        return (y - unscale_bias) / unscale_multiplier


class PerceptronWithNormalNoise(ModelNN):
    r"""Sample from :math:`F^{-1}\\left(\\mathcal{N}(f_{\\theta}(x), \\sigma^2)\\right)`, where :math:`\\sigma` is the standard deviation of the noise, :math:`f_{\\theta}(x)` is perceptron with weights :math:`\\theta`, and :math:`F` is the linear transformation from `bounds` to [-1, 1]."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        leaky_relu_coef: float,
        output_bounds: Union[List[Any], np.array],
        sigma: float,
        normalize_output_coef: float,
        weight_min: Optional[float] = None,
        weight_max: Optional[float] = None,
    ):
        r"""Instantiate PerceptronWithNormalNoise.

        :param dim_input: Dimensionality of input (x)
        :type dim_input: int
        :param dim_output: Dimensionality of output :math `f_{\\theta}(x)`
        :type dim_output: int
        :param dim_hidden: Dimensionality of hidden layers in perceptron :math `f_{\\theta}(x)`
        :type dim_hidden: int
        :param n_hidden_layers: Number of hidden layers in perceptron :math `f_{\\theta}(x)`
        :type n_hidden_layers: int
        :param leaky_relu_coef: Negative slope of the nn.LeakyReLU in perceptron.
        :type leaky_relu_coef: float
        :param output_bounds: Bounds for the output
        :type output_bounds: Union[List[Any], np.array]
        :param sigma: Standard deviation of normal distribution
        :type sigma: float
        :param normalize_output_coef: Coefficient :math `L` in latest activation function in perceptron :math `(1 - 3 \\sigma)\\tanh\\left(\\frac{\\cdot}{L}\\right)`. We use :math `3\\sigma` rule here to guarantee that sampled random variable is in [-1, 1] with good probability. Moreover, :math `L` is an hyperparameter that stabilizes the training in small times.
        :type normalize_output_coef: float
        :param weight_min: Minimum value for weight. If `None` the weights are not clipped, defaults to None
        :type weight_min: Optional[float], optional
        :param weight_max: Maximum value for weight. If `None` the weights are not clipped, defaults to None
        :type weight_max: Optional[float], optional
        """
        super().__init__()
        self.std = sigma
        self.normalize_output_coef = normalize_output_coef

        self.perceptron = ModelPerceptron(
            dim_input=dim_input,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            n_hidden_layers=n_hidden_layers,
            leaky_relu_coef=leaky_relu_coef,
            weight_min=weight_min,
            weight_max=weight_max,
        )

        self.bounds_handler = BoundsHandler(output_bounds)

        self.register_parameter(
            name="__scale_tril_matrix",
            param=torch.nn.Parameter(
                (self.std * torch.eye(dim_output)).float(),
                requires_grad=False,
            ),
        )
        self.cache_weights()

    def get_mean(self, observations):
        assert 1 - 3 * self.std > 0, "1 - 3 std should be greater than 0"
        # We should guarantee with good probability that sampled actions are within action bounds that are scaled to [-1, 1]
        # That is why we use 3 sigma rule here
        return (1 - 3 * self.std) * torch.tanh(
            self.perceptron(observations) / self.normalize_output_coef
        )

    def forward(self, observations):
        return self.bounds_handler.unscale_from_minus_one_one_to_bounds(
            self.get_mean(observations)
        )

    def log_pdf(self, observations, actions):
        means = self.get_mean(observations)
        scaled_actions = self.bounds_handler.scale_from_bounds_to_minus_one_one(actions)

        return MultivariateNormal(
            loc=means,
            scale_tril=self.get_parameter("__scale_tril_matrix"),
        ).log_prob(scaled_actions)

    def sample(self, observation):
        mean = self.get_mean(observation)
        sampled_scaled_action = MultivariateNormal(
            loc=mean,
            scale_tril=self.get_parameter("__scale_tril_matrix"),
        ).sample()

        sampled_action = self.bounds_handler.unscale_from_minus_one_one_to_bounds(
            sampled_scaled_action
        )

        return sampled_action