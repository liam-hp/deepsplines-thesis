import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath('../DeepSplines'))
from deepsplines.ds_modules import dsnn

class LinearReLU(nn.Module):
  def __init__(self, layers, input_size=8):
    super(LinearReLU, self).__init__()

    self.layers = nn.Sequential()  
    prev=input_size
    for l in layers:
      self.layers.append(nn.Linear(prev, l))
      self.layers.append(nn.ReLU())
      prev=l
    self.layers.append(nn.Linear(prev, 1))

    self.num_params = sum(p.numel() for p in self.parameters())

  def forward(self, x):
    x = self.layers(x)
    return x

class LinearBSpline(dsnn.DSModule):
  def __init__(self, layers, s=5, r=4, init="relu", input_size=8):
    super(LinearBSpline, self).__init__()
    self.fc_ds = nn.ModuleList()
    opt_params = {'size': s,  'range_': r, 'init': init, 'save_memory': False}
    self.layers = nn.Sequential()

    prev=input_size
    for l in layers:
      self.layers.append(nn.Linear(prev, l))
      self.layers.append(dsnn.DeepBSpline('fc', l, **opt_params))
      prev=l
    self.layers.append(nn.Linear(prev, 1))

    self.num_params = self.get_num_params()
  
  def get_layers(self):
    return self.layers
  
  def forward(self, x):
    return self.layers(x)
  


import torch
import torch.nn as nn

# a simple wrapper around DeepBSpline_Func
class SingleBSpline(nn.Module):
    def __init__(self, size, grid_value, init="relu", save_memory=False):
        """
        Args:
            size (int): Number of spline knots (control points).
            grid_value (float): Knot spacing.
            save_memory (bool): Toggle memory-efficient vs. standard mode.
        """
        super().__init__()
        self.size = size
        self.save_memory = save_memory
        
        # One-dimensional set of coefficients:
        # shape = (size,)
        if(init=="relu"):
          self.coefficients = nn.Parameter(
            torch.tensor([0.0,0.0,1.0], requires_grad=True)
          )
        else:
          self.coefficients = nn.Parameter(
              torch.randn(size, requires_grad=True)
          )

        self.coefficients_vect = self.coefficients.view(-1)
        
        # "grid" can be a constant tensor or Parameter
        self.grid = torch.tensor(grid_value, requires_grad=False)
        
        # For a single B-spline, the "zero_knot_index" can be just 0
        # (or precompute if needed). We'll store it as a tensor for consistency.
        self.zero_knot_index = torch.tensor(0, dtype=torch.float)

    def forward(self, x):
        """
        x can be of any shape (batch_size, *, H, W, ...). 
        DeepBSpline_Func expects:
          - x in shape (batch_size, num_activations, H, W)
          - coefficients_vect in shape (num_activations * size,)
            But since we have only 1 set of coefficients, we can treat
            num_activations = 1 and do a simple reshape.

        We'll just expand dims so x has the 'channel' dimension of 1,
        if it doesn't already.
        """

        return DeepBSpline_Func.apply(
            x, 
            self.coefficients_vect, 
            self.grid, 
            self.zero_knot_index,  # single value
            self.size,
            self.save_memory
        )
    def get_coeffs_vect(self):
       return self.coefficients_vect


# copied from deepSplines
class DeepBSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.

    If save_memory=True, use a memory efficient version at the expense of
    additional running time. (see module's docstring for details)
    """
    
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size,
                save_memory):

        # First, we clamp the input to the range
        # [leftmost coefficient, second righmost coefficient].
        # We have to clamp, on the right, to the second righmost coefficient,
        # so that we always have a coefficient to the right of x_clamped to
        # compute its output. For the values outside the range,
        # linearExtrapolations will add what remains to compute the final
        # output of the activation, taking into account the slopes
        # on the left and right.
        x_clamped = x.clamp(min=-(grid.item() * (size // 2)),
                            max=(grid.item() * (size // 2 - 1)))

        floored_x = torch.floor(x_clamped / grid)  # left coefficient
        fracs = x_clamped / grid - floored_x  # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)

        ctx.save_memory = save_memory

        if save_memory is False:
            ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)
        else:
            ctx.size = size
            ctx.save_for_backward(x, coefficients_vect, grid,
                                  zero_knot_indexes)

            # compute leftmost and rightmost slopes for linear extrapolations
            # outside B-spline range
            num_activations = x.size(1)
            coefficients = coefficients_vect.view(num_activations, size)
            leftmost_slope = (coefficients[:, 1] - coefficients[:, 0])\
                .div(grid).view(1, -1, 1, 1)
            rightmost_slope = (coefficients[:, -1] - coefficients[:, -2])\
                .div(grid).view(1, -1, 1, 1)

            # peform linear extrapolations outside B-spline range
            leftExtrapolations = (x.detach() + grid * (size // 2))\
                .clamp(max=0) * leftmost_slope
            rightExtrapolations = (x.detach() - grid * (size // 2 - 1))\
                .clamp(min=0) * rightmost_slope
            # linearExtrapolations is zero for inputs inside B-spline range
            linearExtrapolations = leftExtrapolations + rightExtrapolations

            # add linear extrapolations to B-spline expansion
            activation_output = activation_output + linearExtrapolations

        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        save_memory = ctx.save_memory

        if save_memory is False:
            fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        else:
            size = ctx.size
            x, coefficients_vect, grid, zero_knot_indexes = ctx.saved_tensors

            # compute fracs and indexes again (do not save them in ctx)
            # to save memory
            x_clamped = x.clamp(min=-(grid.item() * (size // 2)),
                                max=(grid.item() * (size // 2 - 1)))

            floored_x = torch.floor(x_clamped / grid)  # left coefficient
            # distance to left coefficient
            fracs = x_clamped / grid - floored_x

            # This gives the indexes (in coefficients_vect) of the left
            # coefficients
            indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        grad_x = (coefficients_vect[indexes + 1] -
                  coefficients_vect[indexes]) / grid * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        if save_memory is True:
            # Add gradients from the linear extrapolations
            tmp1 = ((x.detach() + grid * (size // 2)).clamp(max=0)) / grid
            grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                                (-tmp1 * grad_out).view(-1))
            grad_coefficients_vect.scatter_add_(0,
                                                indexes.view(-1) + 1,
                                                (tmp1 * grad_out).view(-1))

            tmp2 = ((x.detach() - grid * (size // 2 - 1)).clamp(min=0)) / grid
            grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                                (-tmp2 * grad_out).view(-1))
            grad_coefficients_vect.scatter_add_(0,
                                                indexes.view(-1) + 1,
                                                (tmp2 * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None

