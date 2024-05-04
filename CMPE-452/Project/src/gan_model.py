import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Function
import math
from torch import nn
from torch import Tensor
from torch.nn import functional as F



# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('euclid') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias == True:
            nn.init.constant_(m.b.data, 0)


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False), # in_channels = nz, out_channels = ngf * 8, kernel_size = 4, stride=1, padding=0, bias=True
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.main_list = list(self.main.children())
        self.stage1 = nn.Sequential(*self.main_list[:1])
        self.stage2 = nn.Sequential(*self.main_list[2:4])
        self.stage3 = nn.Sequential(*self.main_list[5:7])
        self.stage4 = nn.Sequential(*self.main_list[8:10])
        self.stage5 = nn.Sequential(*self.main_list[11:12])

    def forward(self, input):
         # input is ``(nc) x 64 x 64``
        testi = input.shape
        stage1 = self.stage1(input)
        test1 = stage1.shape
         # state size. ``(ndf) x 32 x 32``
        stage2 = self.stage2(stage1)
        test2 = stage2.shape
        # state size. ``(ndf*2) x 16 x 16``
        stage3 = self.stage3(stage2)
        test3 = stage3.shape
        # state size. ``(ndf*4) x 8 x 8``
        stage4 = self.stage4(stage3)
        test4 = stage4.shape
         # state size. ``(ndf*8) x 4 x 4``
        stage5 = self.stage5(stage4)
        test5 = stage5.shape
         # state size. ``1?``
        return self.main(input)


class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)
        #output = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1)
        output = W_col.unsqueeze(2)
        test = X_col.unsqueeze(0)
        test = torch.cdist(W_col, X_col, p=2)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col
    
class adder_t(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)
        output = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1)
        output = W_col.unsqueeze(2)
        output = -(torch.square(W_col.unsqueeze(2))/2)-(torch.square(X_col.unsqueeze(2))/2)+W_col*X_col
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col

def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    out = adder.apply(W_col,X_col)
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out

def adder2d_t_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - 1) * stride - 2 * padding + (h_filter - 1) + 1
    w_out = (w_x - 1) * stride - 2 * padding + (w_filter - 1) + 1
    padding = ((h_out - 1)*stride+h_filter-h_x)/2
    padding = int(padding)
    #h_out = (h_x - h_filter + 2 * padding) / stride + 1
    #w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    out = adder_t.apply(W_col,X_col)
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out

class adder2d(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, bias = False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x,self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output
    
class adder2d_t(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, bias = False):
        super(adder2d_t, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_t_function(x,self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output


class euclidconv(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(euclidconv,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        self.homotopy = 1

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(input,
                      self._reversed_padding_repeated_twice,
                      mode=self.padding_mode), weight, self.bias, self.stride,
                _pair(0), self.dilation, self.groups)
        out = F.conv2d(input, weight, self.bias, self.stride, self.padding,
                       self.dilation, self.groups)

        return out

    def forward(self, input: Tensor) -> Tensor:
        mult_value = self._conv_forward(input, self.weight)

        n_batch, _, h, w = mult_value.shape
        c_out, _, _, _ = self.weight.shape
        weight_squared = (self.weight.view(c_out, -1)**2).sum(dim=-1)
        weight_squared_expand = weight_squared.repeat(
            (n_batch, h, w, 1)).transpose(1, 3)

        input_squared = self._conv_forward(input**2,
                                           torch.ones_like(self.weight))

        out = mult_value - 0.5 * self.homotopy * (weight_squared_expand +
                                                  input_squared)

        return out

class euclidconv_t(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(euclidconv_t,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.homotopy = 1

    def _conv_forward(self, input, weight):
        num_spatial_dims = 2
        output_size = None
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        out = F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        
        return out


    def forward(self, input: Tensor) -> Tensor:
        mult_value = self._conv_forward(input, self.weight)

        n_batch, _, h, w = mult_value.shape
        _, c_out, _, _ = self.weight.shape
        weight_squared = (self.weight.view(c_out, -1)**2).sum(dim=-1)
        weight_squared_expand = weight_squared.repeat(
            (n_batch, h, w, 1)).transpose(1, 3)

        input_squared = self._conv_forward(input**2,
                                           torch.ones_like(self.weight))

        out = mult_value - 0.5 * self.homotopy * (weight_squared_expand +
                                                  input_squared)

        return out

class Euclid_good(Function):
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)
        test_output = -(1/2)*X_col.unsqueeze(0)**2-(1/2)*W_col.unsqueeze(2)**2+X_col.unsqueeze(0)*W_col.unsqueeze(2)
        test_output = test_output.sum(1)
        output = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))**2).sum(1)
        return (-1/2)*output
    
    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        # grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        #test = 1/0 # For debugging purposes
        return grad_W_col, grad_X_col

class euclid(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        output = -(torch.square(W_col.unsqueeze(2))/2)-(torch.square(X_col.unsqueeze(2))/2)+W_col*X_col
        ctx.save_for_backward(W_col,X_col,output)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col
    
class euclid_t(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)
        output = torch.cdist(W_col, X_col, p=2)
        #output = -(torch.square(W_col.unsqueeze(2))/2)-(torch.square(X_col.unsqueeze(2))/2)+W_col*X_col
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col,output = ctx.saved_tensors
        grad_W_col = torch.cdist(X_col, output) * grad_output
        grad_X_col = torch.cdist(W_col, output) * grad_output
        return grad_W_col, grad_X_col


def euclid2d_t_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - 1) * stride - 2 * padding + (h_filter - 1) + 1
    w_out = (w_x - 1) * stride - 2 * padding + (w_filter - 1) + 1
    padding = ((h_out - 1)*stride+h_filter-h_x)/2
    padding = int(padding)

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    out = Euclid_good.apply(W_col,X_col)
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out

def euclid2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    out = Euclid_good.apply(W_col,X_col)
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out


class euclid2d(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, bias = False):
        super(euclid2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = euclid2d_function(x, self.weight, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output
    
class euclid2d_t(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, bias = False):
        super(euclid2d_t, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = euclid2d_t_function(x,self.weight, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output


def convnxn(in_planes, out_planes, kernel, stride, padding, bias):
    " nxn convolution with padding "
    return euclidconv(in_planes, out_planes, kernel, stride=stride, padding=padding, bias=bias) # in_channels = nz, out_channels = ngf * 8, kernel_size = 4, stride=1, padding=0, bias=True

def convnxn_t(in_planes, out_planes, kernel, stride, padding, bias):
    " nxn transposed convolution with padding "
    return euclidconv_t(in_planes, out_planes, kernel, stride=stride, padding=padding, bias=bias) # in_channels = nz, out_channels = ngf * 8, kernel_size = 4, stride=1, padding=0, bias=True
 

class Discriminator_Euclid(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator_Euclid, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            convnxn(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            convnxn(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            convnxn(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            convnxn(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            convnxn(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        #self.main_list = list(self.main.children())
        #self.stage1 = nn.Sequential(*self.main_list[:1])
        #self.stage2 = nn.Sequential(*self.main_list[2:4])
        #self.stage3 = nn.Sequential(*self.main_list[5:7])
        #self.stage4 = nn.Sequential(*self.main_list[8:10])
        #self.stage5 = nn.Sequential(*self.main_list[11:12])

    def forward(self, input):
        '''
        # input is ``(nc) x 64 x 64``
        testi = input.shape
        stage1 = self.stage1(input)
        test1 = stage1.shape
         # state size. ``(ndf) x 32 x 32``
        stage2 = self.stage2(stage1)
        test2 = stage2.shape
        # state size. ``(ndf*2) x 16 x 16``
        stage3 = self.stage3(stage2)
        test3 = stage3.shape
        # state size. ``(ndf*4) x 8 x 8``
        stage4 = self.stage4(stage3)
        test4 = stage4.shape
         # state size. ``(ndf*8) x 4 x 4``
        stage5 = self.stage5(stage4)
        test5 = stage5.shape
         # state size. ``1 x 1 x 1``'''
        return self.main(input)
    

class Generator_Euclid(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator_Euclid, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            convnxn_t( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            convnxn_t(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            convnxn_t( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            convnxn_t( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            convnxn_t( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )
        #self.main_list = list(self.main.children())
        #self.stage1 = nn.Sequential(*self.main_list[:2])
        #self.stage2 = nn.Sequential(*self.main_list[3:5])
        #self.stage3 = nn.Sequential(*self.main_list[6:8])
        #self.stage4 = nn.Sequential(*self.main_list[9:11])
        #self.stage5 = nn.Sequential(*self.main_list[12:13])

    def forward(self, input):
        '''
        # input is Z, going into a convolution
        testi = input.shape
        stage1 = self.stage1(input)
        test1 = stage1.shape
        # state size. ``(ngf*8) x 4 x 4``
        stage2 = self.stage2(stage1)
        test2 = stage2.shape
        # state size. ``(ngf*4) x 8 x 8``
        stage3 = self.stage3(stage2)
        test3 = stage3.shape
        # state size. ``(ngf*2) x 16 x 16``
        stage4 = self.stage4(stage3)
        test4 = stage4.shape
        # state size. ``(ngf) x 32 x 32``
        stage5 = self.stage5(stage4)
        test5 = stage5.shape
        # state size. ``(nc) x 64 x 64``'''
        return self.main(input)