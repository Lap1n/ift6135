import torch

class DropoutFunctionHomeMade(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p):
        random_activation = (torch.rand(input.size()) < p).type(torch.cuda.FloatTensor)
        ctx.random_activation = random_activation
        return torch.mul(input, random_activation)

    @staticmethod
    def backward(ctx, grad_output):
        random_activation =  ctx.random_activation
        return torch.mul(grad_output, random_activation), None


class DropoutHomeMade(torch.nn.Module):
    def __init__(self, p):
        super(DropoutHomeMade, self).__init__()
        self.p = p

    def forward(self, input):
        if self.train():
            return DropoutFunctionHomeMade.apply(input, self.p)
        else:
            return torch.mul(input, self.p)
