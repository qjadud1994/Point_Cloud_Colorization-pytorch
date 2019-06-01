import torch
import torch.nn as nn
from torch.autograd import Variable

# Conv2d : (N, C_in, H, W)
# conv2d : (N, C_in, D, H, W)

class Generator(nn.Module):

    def __init__(self, num_pts, drop_p=0.5):
        super(Generator, self).__init__()

        self.num_pts = num_pts

        # input : [bs, 3, N, 1]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 1), stride=1, padding=(1, 0))  # [bs, 64, N, 1]
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0) # [bs, 128, N, 1]
        self.conv2_bn = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0) # [bs, 128, N, 1]
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0) # [bs, 512, N, 1]
        self.conv4_bn = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0) # [bs, 1024, N, 1]
        self.conv5_bn = nn.BatchNorm2d(1024)

        self.max_pool = nn.MaxPool2d(kernel_size=(self.num_pts, 1), stride=(2, 2), padding=0)  # [bs, 1024, 1, 1]

        # expand : [bs, 1024, N, 1]
        # concat : [bs, 2880, N, 1]

        self.conv6 = nn.Conv2d(2880, 1024, kernel_size=1, stride=1, padding=0) # [bs, 1024, N, 1]
        self.conv6_bn = nn.BatchNorm2d(1024)

        self.conv7 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0) # [bs, 256, N, 1]
        self.conv7_bn = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # [bs, 128, N, 1]
        self.conv8_bn = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0) # [bs, 3, N, 1]

        # reshape : [bs, 3, N]

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_p)
        self.tanh = nn.Tanh()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):

        input = input.unsqueeze(-1)

        c1 = self.relu(self.conv1_bn(self.conv1(input)))
        c2 = self.relu(self.conv2_bn(self.conv2(c1)))
        c3 = self.relu(self.conv3_bn(self.conv3(c2)))
        c4 = self.relu(self.conv4_bn(self.conv4(c3)))
        c5 = self.relu(self.conv5_bn(self.conv5(c4)))

        mp = self.max_pool(c5)

        expand = mp.repeat(1, 1, self.num_pts, 1)

        concat = torch.cat([expand, c1, c2, c3, c4, c5], dim=1)  # batch_size x (2048 + 256 + 512 + 64) x N x 1

        c6 = self.relu(self.conv6_bn(self.conv6(concat)))
        c6 = self.dropout(c6)

        c7 = self.relu(self.conv7_bn(self.conv7(c6)))
        c7 = self.dropout(c7)

        c8 = self.relu(self.conv8_bn(self.conv8(c7)))

        out = self.tanh(self.conv9(c8))
        out = out.squeeze(3)

        return out


class Discriminator(nn.Module):
    # initializers
    def __init__(self, num_pts):
        super(Discriminator, self).__init__()

        self.num_pts = num_pts

        # input : [bs, 6, N, 1]
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(5, 1), stride=1, padding=(2, 0))  # [bs, 64, N, 1]
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0) # [bs, 128, N, 1]
        self.conv2_bn = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0) # [bs, 128, N, 1]
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0) # [bs, 512, N, 1]
        self.conv4_bn = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0) # [bs, 2048, N, 1]
        self.conv5_bn = nn.BatchNorm2d(2048)

        self.max_pool = nn.MaxPool2d(kernel_size=(self.num_pts, 1), stride=(2, 2), padding=0)  # [bs, 2048, 1, 1]

        # reshape : [bs, 2048]
        self.dense1 = nn.Linear(2048 , 256) # [bs, 256]
        self.dense1_bn = nn.BatchNorm1d(256)

        self.dense2 = nn.Linear(256 , 256) # [bs, 256]
        self.dense2_bn = nn.BatchNorm1d(256)

        self.dense3 = nn.Linear(256 , 1) # [bs, 1]

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(0.7)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, pt, color):

        x = torch.cat([pt, color], 1)
        x = x.unsqueeze(-1)

        x = self.relu(self.conv1_bn(self.conv1(x)))
        x = self.relu(self.conv2_bn(self.conv2(x)))
        x = self.relu(self.conv3_bn(self.conv3(x)))
        x = self.relu(self.conv4_bn(self.conv4(x)))
        x = self.relu(self.conv5_bn(self.conv5(x)))

        x = self.max_pool(x)

        #x = x.squeeze()
        x = x.view(x.size(0), -1)

        x = self.relu(self.dense1_bn(self.dense1(x)))
        x = self.relu(self.dense2_bn(self.dense2(x)))

        x = self.dropout(x)
        x = self.sigmoid(self.dense3(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def test():
    G = Generator(num_pts=2048)
    D = Discriminator(num_pts=2048)

    G.cuda()
    G.train()
    D.cuda()
    D.train()

    input_g = Variable(torch.rand(32, 3, 2048)).cuda()
    G_out = G(input_g)
    print("G output : ", G_out.size())

    print()
    input_d = Variable(torch.rand(32, 3, 2048)).cuda()
    D_out = D(input_d, G_out)
    print("G output : ", D_out.size())

#test()

