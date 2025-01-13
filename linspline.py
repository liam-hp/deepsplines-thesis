import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.abspath('../DeepSplines'))
from deepsplines.ds_modules.deepBspline import DeepBSpline
import deepsplines


class LSplineFromBSpline(nn.Module):
    def __init__(self, bspline_layers):
        super(LSplineFromBSpline, self).__init__()

        self.layers = nn.Sequential()
        for layer in bspline_layers:
            if(type(layer) is deepsplines.ds_modules.deepBspline.DeepBSpline):
                layer_locs = layer.grid_tensor.detach()
                layer_coeffs = layer.coefficients_vect.view(layer.num_activations, layer.size).detach()
                self.layers.append(LinearSplineLayer(layer_locs, layer_coeffs))
            else:
                self.layers.append(layer)


    def get_layers(self):
        return self.layers

    def forward(self, x):
        x = self.layers(x)
        return x

class LinearSplineLayer(nn.Module):

    def __init__(self, locs, coeffs):
        super(LinearSplineLayer, self).__init__()
        # Assume locs and coeffs are lists of tensors

        self.locs = nn.Parameter(locs.contiguous())
        self.coeffs = nn.Parameter(coeffs.contiguous())
        # self.num_knots = self.locs.size(0) # pre-computing to save time

    def forward(self, x):

        # x is (batch_size, num_splines)
        group_by_splines = x.transpose(0, 1)  # Shape: (num_splines, batch_size)
        
        idx = torch.searchsorted(self.locs, group_by_splines, right=False) - 1 # get the spline indices for each input
        idx = idx.clamp(0, self.locs.size(1) - 2) # make sure we stay in bounds
        
        # spline_indices = torch.arange(self.num_knots, device=x.device).unsqueeze(1) 
        
        # x0 and x1 represent the left and right boundaries of the spline segments for each x in the batch.

        # x0 = self.locs[spline_indices, idx]
        # x1 = self.locs[spline_indices, idx + 1]

        x0 = torch.gather(self.locs, 1, idx)
        x1 = torch.gather(self.locs, 1, idx+1)

        # y0 and y1 are the corresponding coefficients at x0 and x1.

        # y0 = self.coeffs[spline_indices, idx]
        # y1 = self.coeffs[spline_indices, idx + 1]

        y0 = torch.gather(self.coeffs, 1, idx)
        y1 = torch.gather(self.coeffs, 1, idx + 1)
        
        # 7 flops:
        t = (group_by_splines - x0) / (x1 - x0 + 1e-6) # 1e-6 to prevent division by zero
        out = y0 + t * (y1 - y0)
        
        return out.transpose(0, 1)  # Shape: (batch_size, num_splines)

    def get_flops(self):
        return 7 * len(self.coeffs) * len(self.coeffs[0])
        # 7 * number of splines * len(self.coeffs)

'''

#! Register buffer so they're not considered when computing

self.register_buffer("locs", locs)
self.register_buffer("coeffs", coeffs)

#! torch.gather?

'''


class LinearSpline():
    def __init__(self, locs, coeffs):
        super(LinearSpline, self).__init__()
        self.locs = locs
        self.coeffs = coeffs
        
        return

    def forward(self, x):

        raise NotImplementedError

    def get_locs_coeffs(self):
        return (self.locs, self.coeffs)
    

'''

Other LSpline Implementations


    my implementation:

    def __init__(self, locs, coeffs, mode="fc"):

        print("checking reload2")

        super(LinearSplineLayer, self).__init__()
        
        if mode != "fc":
            raise NotImplementedError("Only fully connected (fc) mode is implemented")
        
        self.locs = [loc.float() for loc in locs]
        self.coeffs = [coeff.float() for coeff in coeffs]
    
    
    def forward(self, x):

        # x is grouped by input, where each input consists of [forspline1, forspline2, forspline3, etc.]
        # lets transpose so we group by splines to then process through in one loop
        group_by_splines = torch.transpose(x, 0, 1)

        # initialize our output array to save
        out = torch.zeros_like(group_by_splines, dtype=torch.float32)
        
        for spline in range(len(self.locs)):
            
            g = group_by_splines[spline]

            locs = self.locs[spline]
            coeffs = self.coeffs[spline]

            # Find the interval indices
            idx = torch.searchsorted(locs, g) - 1
            idx = torch.clamp(idx, 0, len(locs) - 2)

            # Get the surrounding points
            x0, x1 = locs[idx], locs[idx + 1]
            y0, y1 = coeffs[idx], coeffs[idx + 1]

            # Compute the interpolation factor
            t = (g - x0) / (x1 - x0)
            
            # Linear interpolation
            out[spline] = y0 + t * (y1 - y0)
                           
        return torch.transpose(out, 0, 1)

    GPT (works and fast):
        def forward(self, x):
            # Transpose to group by splines
            group_by_splines = x.transpose(0, 1)  # Shape: (num_splines, batch_size)
            
            # Stack locs and coeffs for batch processing
            locs = torch.stack(self.locs)          # Shape: (num_splines, num_locs)
            coeffs = torch.stack(self.coeffs)      # Shape: (num_splines, num_coeffs)
            
            # Find interval indices for all splines at once
            idx = torch.searchsorted(locs, group_by_splines, right=False) - 1
            idx = idx.clamp(0, locs.size(1) - 2)
            
            # Gather x0, x1, y0, y1 using advanced indexing
            batch_indices = torch.arange(locs.size(0)).unsqueeze(1).to(x.device)
            x0 = locs[batch_indices, idx]
            x1 = locs[batch_indices, idx + 1]
            y0 = coeffs[batch_indices, idx]
            y1 = coeffs[batch_indices, idx + 1]
            
            # Compute interpolation factor and interpolate
            t = (group_by_splines - x0) / (x1 - x0)
            out = y0 + t * (y1 - y0)
            
            return out.transpose(0, 1)
'''