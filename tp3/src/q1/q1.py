import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


class BaseModel(nn.Module):
    def __init__(self, input_size):
        super(BaseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)


class JSDModel(nn.Module):
    def __init__(self, input_size):
        super(JSDModel, self).__init__()
        self.model = nn.Sequential(BaseModel(input_size), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


def compute_jsd(D, x, y):
    return torch.log(torch.Tensor([2])) + torch.log(D(x)).mean() / 2 + torch.log(1 - D(y)).mean() / 2


def JensenShannonDivergence(p, q, n_mini_batch=1000):
    x = torch.Tensor(next(p))
    y = torch.Tensor(next(q))
    input_size = x.size()[1]
    D = JSDModel(input_size)
    # optimizer = optim.SGD(D.parameters(), lr=1e-3)
    optimizer = optim.Adam(D.parameters())
    for mini_batch in range(n_mini_batch):
        D.zero_grad()
        jsd = compute_jsd(D, x, y)
        jsd.backward(torch.FloatTensor([-1]))
        optimizer.step()
    return D, compute_jsd(D, x, y)


JSD = JensenShannonDivergence


class WDModel(nn.Module):
    def __init__(self, input_size):
        super(WDModel, self).__init__()
        self.model = nn.Sequential(BaseModel(input_size), nn.ReLU())

    def forward(self, x):
        return self.model(x)


def compute_wd(T, x, y):
    return T(x).mean() - T(y).mean()


def compute_gradient_penality(T, x, y, lambda_grad_penality):
    a = torch.rand(x.size()[0],1)
    a.expand_as(x)
    a.requires_grad = True
    z = a * x + (1 - a) * y
    T_z = T(z)
    grads = autograd.grad(outputs=T_z,
                          inputs=z,
                          grad_outputs=torch.ones(T_z.size()),
                          create_graph=True,
                          retain_graph=True)[0]
    gradient_penalty = lambda_grad_penality * ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def WassersteinDistance(p, q, n_mini_batch=1000, lambda_grad_penality=10):
    x = torch.Tensor(next(p))
    y = torch.Tensor(next(q))
    input_size = x.size()[1]
    T = BaseModel(input_size)
    optimizer = optim.Adam(T.parameters())
    for mini_batch in range(n_mini_batch):
        T.zero_grad()
        wd = compute_wd(T, x, y)
        total_loss = wd - compute_gradient_penality(T, x, y, lambda_grad_penality)
        total_loss.backward(torch.FloatTensor([-1]))
        optimizer.step()
    return T, compute_wd(T, x, y) - compute_gradient_penality(T, x, y, lambda_grad_penality)


WD = WassersteinDistance

if __name__ == "__main__":
    from tp3.src.given_code.samplers import distribution1
    import numpy as np
    import matplotlib.pyplot as plt

    batch_size = 512
    n_mini_batch = 1000
    phi = 1
    p = iter(distribution1(0, batch_size))
    q = iter(distribution1(phi, batch_size))
    D, jsd = JSD(p, q, n_mini_batch)
    T, wd  =  WD(p, q, n_mini_batch, lambda_grad_penality=10)
    print(f"JSD : got={jsd.item()}, expected={np.log(2)}")
    print(f"WD : got={wd.item()}, expected={phi}")

    ## q1.3
    batch_size = 512
    n_mini_batch = 500
    lambda_grad_penality = 10
    computed_jsd = []
    computed_wd = []
    phis = np.linspace(-1, 1, 21)
    for phi in phis:
        p = iter(distribution1(0))
        q = iter(distribution1(phi))
        D, jsd = JSD(p, q, n_mini_batch)
        computed_jsd.append(jsd)
        T, wd = WD(p, q, n_mini_batch, lambda_grad_penality)
        computed_wd.append(wd)

    plt.plot(phis, computed_jsd, label="JSD")
    plt.plot(phis, computed_wd, label="WD")
    plt.legend()
    plt.savefig('q1.3.png')
    plt.show()
