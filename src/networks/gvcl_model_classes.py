import math
import numpy as np
from functools import partial

import torch
from torch import nn
from torch import distributions
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

device = "cuda:0"


class MultiHeadFiLMCNN(nn.Module):
    def __init__(self, input_shape, conv_sizes, fc_sizes, output_dims, film_type="point", single_head=False, global_avg_pool=False, prior_vars=[0.0], init_var=-7.0, gauss_mixture=1, tau=1.0):
        assert gauss_mixture == len(prior_vars)
        super().__init__()
        self.input_shape = input_shape
        self.conv_sizes = conv_sizes
        self.fc_sizes = fc_sizes
        self.output_dims = output_dims
        self.film_type = film_type
        self.single_head = single_head
        self.global_avg_pool = global_avg_pool
        self.prior_vars = prior_vars
        self.init_var = init_var
        self.gauss_mixture = gauss_mixture
        self.tau = tau
        self.construct()

    def construct(self):
        # Num Tasks
        self.num_tasks = len(self.output_dims)

        # Layers
        self.conv_layers = nn.ModuleList([])
        self.conv_film_layers = nn.ModuleList([])

        self.fc_layers = nn.ModuleList([])
        self.fc_film_layers = nn.ModuleList([])

        self.heads = nn.ModuleList([])

        # Films type
        self.set_film_gen_type()

        # Layers
        input_channels = self.input_shape[0]
        input_dimension = self.input_shape[1]

        for conv_size in self.conv_sizes:
            if conv_size == "pool":
                input_dimension = input_dimension // 2
                continue
            output_channels = conv_size[0]
            kernel_size = conv_size[1]
            if len(conv_size) == 2:
                padding = (kernel_size - 1) // 2
            else:
                padding = conv_size[2]

            self.conv_layers.append(MixtureConvLayer(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding, prior_vars=self.prior_vars, init_var=self.init_var, gauss_mixture=self.gauss_mixture, tau=self.tau))
            self.conv_film_layers.append(self.conv_film_gen_type(self.num_tasks, output_channels))

            input_channels = output_channels
            input_dimension = input_dimension + 2 * padding - kernel_size + 1

        if not self.global_avg_pool:
            input_channels *= input_dimension**2

        for hidden_size in self.fc_sizes:
            output_channels = hidden_size
            self.fc_layers.append(MixtureLinearLayer(input_channels, output_channels, prior_vars=self.prior_vars, init_var=self.init_var, gauss_mixture=self.gauss_mixture, tau=self.tau))
            self.fc_film_layers.append(self.fc_film_gen_type(self.num_tasks, output_channels))
            input_channels = output_channels

        if self.single_head:
            self.heads.append(MixtureLinearLayer(input_channels, self.output_dims[0], prior_vars=self.prior_vars, init_var=self.init_var, gauss_mixture=self.gauss_mixture, tau=self.tau))
        else:
            for output_dim in self.output_dims:
                self.heads.append(MixtureLinearLayer(input_channels, output_dim, prior_vars=self.prior_vars, init_var=self.init_var, gauss_mixture=self.gauss_mixture, tau=self.tau))

    def get_task_specific_parameters(self, task):
        modules = nn.ModuleList([self.fc_layers, self.conv_layers, self.fc_film_layers, self.conv_film_layers])
        if self.single_head:
            modules.append(self.heads[0])
        else:
            modules.append(self.heads[task])
        return modules.parameters()

    def set_film_gen_type(self):
        if self.film_type == "point":
            self.conv_film_gen_type = partial(PointFiLMLayer, constant=False, conv=True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant=False)
        elif self.film_type == "scale":
            self.conv_film_gen_type = partial(PointFiLMLayer, constant=False, conv=True, scale_only=True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant=False, scale_only=True)
        elif self.film_type == "bias":
            self.conv_film_gen_type = partial(PointFiLMLayer, constant=False, conv=True, bias_only=True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant=False, bias_only=True)
        elif self.film_type is None:
            self.conv_film_gen_type = partial(PointFiLMLayer, constant=True, conv=True)
            self.fc_film_gen_type = partial(PointFiLMLayer, constant=True)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x, task_labels, num_samples=1, tasks=None):
        batch_size = x.shape[0]
        x = x.repeat([num_samples, 1, 1, 1])

        # Forward
        layer_index = 0
        for conv_size in self.conv_sizes:
            if conv_size == "pool":
                x = F.max_pool2d(x, kernel_size=2, stride=2)
                continue
            x = self.conv_layers[layer_index](x)
            x = self.conv_film_layers[layer_index](x, task_labels, num_samples)
            x = F.relu(x)
            layer_index += 1

        if self.global_avg_pool:
            x = x.view(num_samples, batch_size, x.shape[1], -1).mean(-1)
        else:
            x = x.view(num_samples, batch_size, -1)

        layer_index = 0
        for hidden_size in self.fc_sizes:
            x = self.fc_layers[layer_index](x)
            x = self.fc_film_layers[layer_index](x, task_labels, num_samples)
            x = F.relu(x)
            layer_index += 1

        # Forward heads
        all_tasks = list(range(self.num_tasks))
        outputs = [None for task in all_tasks]
        if tasks is None:
            tasks = all_tasks
        for task in all_tasks:
            head_index = 0 if self.single_head else task
            if task in tasks:
                task_output = self.heads[head_index](x)
                outputs[task] = task_output.reshape([num_samples, batch_size, -1])
            else:
                outputs[task] = torch.zeros(torch.Size([self.output_dims[task]]), device=device)
        return outputs

    def get_kl_task(self, task, lamb=1):
        kl = 0
        for layer in nn.ModuleList([*self.conv_layers, *self.fc_layers]):
            kl += layer.get_kl(lamb)

        if self.single_head:
            kl += self.heads[0].get_kl(lamb)
        else:
            kl += self.heads[task].get_kl(lamb)

        return kl

    def add_task_body_params(self, task):
        for layer in nn.ModuleList([*self.fc_layers, *self.conv_layers]):
            layer.add_new_task()

        if self.single_head:
            self.heads[0].add_new_task()
        else:
            self.heads[task].add_new_task()


class PointFiLMLayer(nn.Module):
    def __init__(self, tasks, width, constant=False, conv=False, scale_only=False, bias_only=False):
        super().__init__()
        self.register_parameter(name="scales", param=Parameter(torch.Tensor(tasks, width), requires_grad=(not constant) and (not bias_only)))
        self.register_parameter(name="shifts", param=Parameter(torch.Tensor(tasks, width), requires_grad=(not constant) and (not scale_only)))
        self.conv = conv
        self.width = width
        self.constant = constant
        init.constant_(self.scales, 1.0)
        init.constant_(self.shifts, 0.0)

    def forward(self, x, task_labels, num_samples):
        scale_values = F.embedding(task_labels, self.scales)
        shift_values = F.embedding(task_labels, self.shifts)
        if self.conv:
            scale_values = scale_values.view(-1, self.width, 1, 1).repeat(num_samples, 1, 1, 1)
            shift_values = shift_values.view(-1, self.width, 1, 1).repeat(num_samples, 1, 1, 1)
        else:
            scale_values = scale_values.view(1, -1, self.width)
            shift_values = shift_values.view(1, -1, self.width)
        return x * scale_values + shift_values


class MixtureLayer(nn.Module):
    def __init__(self, prior_vars=[0.0], init_var=-7.0, gauss_mixture=1, tau=1.0):
        super().__init__()
        self.prior_vars = prior_vars
        self.init_var = init_var
        self.gauss_mixture = gauss_mixture
        self.tau = tau

    def construct(self):
        self.register_parameter(name="W_mean", param=Parameter(torch.Tensor(self.gauss_mixture, *self.weight_shape).to(device)))
        self.register_parameter(name="b_mean", param=Parameter(torch.Tensor(self.gauss_mixture, *self.bias_shape).to(device)))
        self.register_parameter(name="W_var", param=Parameter(torch.Tensor(self.gauss_mixture, *self.weight_shape).to(device)))
        self.register_parameter(name="b_var", param=Parameter(torch.Tensor(self.gauss_mixture, *self.bias_shape).to(device)))
        self.register_parameter(name="W_coff", param=Parameter(torch.Tensor(self.gauss_mixture, *self.weight_shape).to(device)))
        self.register_parameter(name="b_coff", param=Parameter(torch.Tensor(self.gauss_mixture, *self.bias_shape).to(device)))

        for i in range(self.gauss_mixture):
            init.constant_(self.W_var, self.prior_vars[i])
            init.constant_(self.b_var, self.prior_vars[i])
        init.constant_(self.W_coff, 1.0)
        init.constant_(self.b_coff, 1.0)
        for i in range(self.gauss_mixture):
            init.kaiming_uniform_(self.W_mean[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W_mean[i])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.b_mean[i], -bound, bound)

    def add_new_task(self):
        if not hasattr(self, "W_mean"):
            raise Exception(f"Mixture layer {self}: construct!!!")

        self.W_prior_mean = self.W_mean.clone().detach().requires_grad_(False)
        self.b_prior_mean = self.b_mean.clone().detach().requires_grad_(False)
        self.W_prior_var = self.W_var.clone().detach().requires_grad_(False)
        self.b_prior_var = self.b_var.clone().detach().requires_grad_(False)
        self.W_prior_coff = self.W_coff.clone().detach().requires_grad_(False)
        self.b_prior_coff = self.b_coff.clone().detach().requires_grad_(False)

        init.constant_(self.W_var, self.init_var)
        init.constant_(self.b_var, self.init_var)
        init.constant_(self.W_coff, 1.0)
        init.constant_(self.b_coff, 1.0)
        for i in range(self.gauss_mixture):
            init.kaiming_uniform_(self.W_mean[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W_mean[i])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.b_mean[i], -bound, bound)

    def get_kl(self, lamb):
        if not hasattr(self, "W_prior_mean"):
            return torch.tensor(0.0)
        W_kl = upper_bound_kl_divergence_mixture_gauss([self.W_mean, self.W_var, self.W_coff], [self.W_prior_mean, self.W_prior_var, self.W_prior_coff], gauss_mixture=self.gauss_mixture, initial_prior_vars=self.prior_vars, lamb=lamb)
        b_kl = upper_bound_kl_divergence_mixture_gauss([self.b_mean, self.b_var, self.b_coff], [self.b_prior_mean, self.b_prior_var, self.b_prior_coff], gauss_mixture=self.gauss_mixture, initial_prior_vars=self.prior_vars, lamb=lamb)
        return W_kl + b_kl

    def mixture_forward(self, x, weight, bias):
        raise Exception(f"Mixture layer {self}: mixture forward!!!")

    def forward(self, x):
        W_gumbel_softmax = F.gumbel_softmax(self.W_coff, dim=0, hard=False, tau=self.tau)
        W_mean = torch.sum(self.W_mean * W_gumbel_softmax, dim=0)
        W_var = torch.sum(self.W_var * W_gumbel_softmax, dim=0)

        b_gumbel_softmax = F.gumbel_softmax(self.b_coff, dim=0, hard=False, tau=self.tau)
        b_mean = torch.sum(self.b_mean * b_gumbel_softmax, dim=0)
        b_var = torch.sum(self.b_var * b_gumbel_softmax, dim=0)

        output_mean = self.mixture_forward(x, W_mean, b_mean)
        output_var = self.mixture_forward(x**2, torch.exp(W_var), torch.exp(b_var))
        eps = torch.empty(output_var.shape, device=device).normal_(mean=0, std=1)
        return output_mean + torch.sqrt(output_var) * eps


class MixtureConvLayer(MixtureLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", prior_vars=[0.0], init_var=-7.0, gauss_mixture=1, tau=1.0):
        super().__init__(prior_vars, init_var, gauss_mixture, tau)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.weight_shape = torch.Size([out_channels, in_channels // groups, kernel_size, kernel_size])
        self.bias_shape = torch.Size([out_channels])
        self.construct()

    def mixture_forward(self, x, weight, bias):
        if self.padding_mode == "circular":
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2, (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(x, expanded_padding, mode="circular"), weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class MixtureLinearLayer(MixtureLayer):
    def __init__(self, in_features, out_features, bias=True, prior_vars=[0.0], init_var=-7.0, gauss_mixture=1, tau=1.0):
        super().__init__(prior_vars, init_var, gauss_mixture, tau)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weight_shape = torch.Size([out_features, in_features])
        self.bias_shape = torch.Size([out_features])
        self.construct()

    def mixture_forward(self, x, weight, bias):
        return F.linear(x, weight, bias)


def upper_bound_kl_divergence_mixture_gauss(g, p_g, gauss_mixture=1, lamb=1, initial_prior_vars=[0.0]):
    m, p_m = g[0], p_g[0]
    v, p_v = g[1], p_g[1]
    c, p_c = F.softmax(g[2], dim=0), F.softmax(p_g[2], dim=0)
    kl = 0
    for i in range(gauss_mixture):
        kl += torch.sum(c[i] * (torch.log(c[i]) - torch.log(p_c[i])))
        kl += torch.sum(c[i] * compute_kl(m[i], v[i], p_m[i], p_v[i], sum=False, lamb=lamb, initial_prior_var=initial_prior_vars[i]))
    return kl


def compute_kl(mean, exp_var, prior_mean, prior_exp_var, sum=True, lamb=1, initial_prior_var=0.0):
    trace_term = torch.exp(exp_var - prior_exp_var)
    if lamb != 1:
        mean_term = (mean - prior_mean) ** 2 * (lamb * torch.clamp(torch.exp(-prior_exp_var) - (1.0 / np.exp(1.0 * initial_prior_var)), min=0.0) + (1.0 / np.exp(1.0 * initial_prior_var)))
    else:
        mean_term = (mean - prior_mean) ** 2 * torch.exp(-prior_exp_var)
    det_term = prior_exp_var - exp_var

    if sum:
        return 0.5 * torch.sum(trace_term + mean_term + det_term - 1)
    else:
        return 0.5 * (trace_term + mean_term + det_term - 1)
