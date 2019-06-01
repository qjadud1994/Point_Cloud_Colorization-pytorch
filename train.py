import os, time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from datagen import ListDataset
from network import Generator, Discriminator
from util import pc_visualize

""" For learning rate decay """
def adjust_learning_rate(cur_lr, optimizer, gamma, step):
    lr = cur_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

""" set argments """
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/root/DB/')
parser.add_argument('--batch_size', type=int, default=32, help='train batch size')
parser.add_argument('--num_pts', type=int, default=2048, help='number of points 2048/4096')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--dataset', type=str, help='select training dataset')
parser.add_argument('--train_epoch', type=int, default=1000, help='number of train epochs')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--L1_lambda', type=float, default=20, help='lambda for L1 loss')
parser.add_argument('--D_lambda', type=float, default=0.5, help='lambda for D loss')
parser.add_argument('--logdir', default='logs/', type=str, help='Tensorboard log dir')
parser.add_argument('--max_iter', default=100000, type=int, help='Number of training iterations')
parser.add_argument('--show_interval', default=75, type=int, help='interval of showing training conditions')
parser.add_argument('--save_interval', default=5000, type=int, help='interval of save checkpoint models')
parser.add_argument('--save_folder', default='checkpoint/', help='Location to save checkpoint models')
parser.add_argument('--gamma', default=0.5, type=float, help='learning rate decay')
parser.add_argument('--augmentation', type=bool, default=False, help='do augmentation or not')
parser.add_argument('--restore', default=None, type=str,  help='restore from checkpoint')

opt = parser.parse_args()
print(opt)

""" weights save folder & log folder for tensorboard visualization """
if not os.path.exists(opt.save_folder):
    os.mkdir(opt.save_folder)

if not os.path.exists(opt.logdir):
    os.mkdir(opt.logdir)

    
""" initial parameters for training """
epoch = 0
global_iter = 1
step_index = 0
stepvalues = (5000, 10000, 15000) # learning rate decay steps
cur_lrG = opt.lrG
cur_lrD = opt.lrD
    
    
""" Get data loader """
transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# <augmentation=False> for test
trainset = ListDataset(root=opt.root, dataset='densepoint', mode="train", 
                       num_pts=opt.num_pts, transform=transform, 
                       augmentation=opt.augmentation)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=0)


""" Networks : Generator & Discriminator """
G = Generator(opt.num_pts)
D = Discriminator(opt.num_pts)

G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)


""" set CUDA """
G.cuda()
D.cuda()


""" Optimizer """
G_optimizer = optim.Adam(G.parameters(), lr=cur_lrG, betas=(0.9, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=cur_lrD, betas=(0.9, 0.999))


""" Restore """
if opt.restore:
    print('==> Restoring from checkpoint..', opt.restore)
    state = torch.load(opt.restore)
    
    G.load_state_dict(state['G'])
    D.load_state_dict(state['D'])
    G_optimizer.load_state_dict(state["G_optimizer"])
    D_optimizer.load_state_dict(state["D_optimizer"])
    epoch = state["epoch"]
    global_iter += state["iter"]
    cur_lrG = state["lrG"]
    cur_lrD = state["lrD"]
    state = None

    
""" multi-GPU training  """
G = torch.nn.DataParallel(G, device_ids=range(torch.cuda.device_count()))
D = torch.nn.DataParallel(D, device_ids=range(torch.cuda.device_count()))


""" training mode  """
G.train()
D.train()


""" Loss for GAN """
BCE_loss = nn.BCELoss().cuda()
L1_loss = nn.L1Loss().cuda()


""" tensorboard visualize """
writer = SummaryWriter(log_dir=opt.logdir)


""" start training """
print('training start!')

for epoch in range(opt.train_epoch):
    local_iter = 0

    for x, y in train_loader:
        t0 = time.time()

        """ training Discriminator D"""
        D.zero_grad()

        x, y = Variable(x.cuda()), Variable(y.cuda())

        D_real_result = D(x, y).squeeze()
        D_real_loss = BCE_loss(D_real_result, Variable(torch.ones(D_real_result.size()).cuda())) # log(D(x, y))
        D_real_loss *= opt.D_lambda

        D_fake_result = D(x, G(x).detach()).squeeze()
        D_fake_loss = BCE_loss(D_fake_result, Variable(torch.zeros(D_fake_result.size()).cuda()))  # -log(1-D(x, G(x)))
        D_fake_loss *= opt.D_lambda

        D_train_loss = (D_real_loss + D_fake_loss)
        D_train_loss.backward()
        D_optimizer.step()

        """ training generator G """
        G.zero_grad()

        G_result = G(x)
        D_result = D(x, G_result).squeeze()

        G_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))  # log(D(x, G(x))
        l1_loss = opt.L1_lambda * L1_loss(G_result, y)

        G_train_loss = (G_loss + l1_loss)
        G_train_loss.backward()
        G_optimizer.step()

        """ terminal logs """
        t1 = time.time()
        print("|| epoch : %03d || iter : %05d || [%04d/%04d] || G_loss=%.3f || D_loss=%.4f || Time : %.1fms ||" % 
              (epoch, global_iter, local_iter, len(train_loader), G_train_loss.item(), D_train_loss.item(), (t1-t0)*1000), end='\r')

        """ Tensorboard visualization : losses, lr, images """
        if global_iter % opt.show_interval == 0:
            print()

            val_input = x[0].cpu().detach().numpy()
            val_gt = y[0].cpu().detach().numpy()
            val_pred = G_result[0].cpu().detach().numpy()

            val_gt_image = pc_visualize(val_input, val_gt, "GT")
            val_pred_image = pc_visualize(val_input, val_pred, "Pred")

            val_image = torch.ones((3, 480*2, 640))
            val_image[:, :480, :] = val_gt_image
            val_image[:, 480:, :] = val_pred_image

            # tensorboard visualize
            writer.add_scalar('D_train_loss', D_train_loss.item(), global_iter)
            writer.add_scalar('D_fake_loss', D_fake_loss.item(), global_iter)
            writer.add_scalar('D_real_loss', D_real_loss.item(), global_iter)
            writer.add_scalar('G_train_loss', G_train_loss.item(), global_iter)
            writer.add_scalar('G_loss', G_loss.item(), global_iter)
            writer.add_scalar('L1_loss', l1_loss.item(), global_iter)
            writer.add_scalar('step_time', t1-t0, global_iter)
            writer.add_scalar('G_learning_rate', cur_lrG, global_iter)
            writer.add_scalar('D_learning_rate', cur_lrD, global_iter)
            writer.add_image('val_Image', val_image, global_iter)

        """ save checkpoint """
        if global_iter % opt.save_interval == 0 and global_iter > 0:
            print('\nSaving state, iter : %d \n' % global_iter)
            state = {
                'G': G.module.state_dict(),
                "G_optimizer": G_optimizer.state_dict(),
                'D': D.module.state_dict(),
                "D_optimizer": D_optimizer.state_dict(),
                'epoch': epoch,
                'iter': global_iter,
                'lrG' : cur_lrG,
                'lrD' : cur_lrD
            }
            model_file = opt.save_folder + '/ckpt_' + repr(global_iter) + '.pth'
            torch.save(state, model_file)

        """ learning rate decay """
        if global_iter in stepvalues:
            step_index += 1
            cur_lrG = adjust_learning_rate(cur_lrG, G_optimizer, opt.gamma, step_index)
            cur_lrD = adjust_learning_rate(cur_lrD, D_optimizer, opt.gamma, step_index)

        local_iter += 1
        global_iter += 1

        if global_iter > opt.max_iter:
            break

    if global_iter > opt.max_iter:
        break

print("============ Training done ============")
