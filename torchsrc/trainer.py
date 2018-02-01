import datetime
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import scipy.misc
import scipy.io as sio
import nibabel as nib
from scipy.spatial import distance
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import skimage
import random
from utils.image_pool import ImagePool
import torchsrc

def saveOneImg(img,path,cate_name,sub_name,surfix,):
    filename = "%s-x-%s-x-%s.png"%(cate_name,sub_name,surfix)
    file = os.path.join(path,filename)
    scipy.misc.imsave(file, img)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)


def weighted_center(input,threshold=0.75):
    # m= torch.nn.Tanh()
    # input = m(input)

    input = torch.add(input, -input.min().expand(input.size())) / torch.add(input.max().expand(input.size()), -input.min().expand(input.size()))
    m = torch.nn.Threshold(threshold, 0)
    input = m(input)
    # if input.sum()==0:
    #     input=input
    # mask_ind = input.le(0.5)
    # input.masked_fill_(mask_ind, 0.0)
    grid = np.meshgrid(range(input.size()[0]), range(input.size()[1]), indexing='ij')
    x0 = torch.mul(input, Variable(torch.from_numpy(grid[1]).float().cuda())).sum() / input.sum()
    y0 = torch.mul(input, Variable(torch.from_numpy(grid[0]).float().cuda())).sum() / input.sum()
    return x0, y0


# def max_center(input,target,pts):
#     input.max()
#     return x0, y0


def get_distance(target,score,ind,Threshold=0.75):
    dist_list = []
    coord_list = []
    target_coord_list = []
    weight_coord_list = []
    for i in range(target.size()[1]):
        targetImg = target[ind,i,:,:].data.cpu().numpy()
        scoreImg = score[ind,i,:,:].data.cpu().numpy()
        targetCoord = np.unravel_index(targetImg.argmax(),targetImg.shape)
        scoreCoord = np.unravel_index(scoreImg.argmax(),scoreImg.shape)
        # grid = np.meshgrid(range(score.size()[2]), range(score.size()[3]), indexing='ij')
        # x0 = torch.mul(score[ind, i, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum() / score[ind, i, :,
        #                                                                                               :].sum()
        # y0 = torch.mul(score[ind, i, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum() / score[ind, i, :,
        #                                                                                               :].sum()
        #
        y0,x0 = weighted_center(score[ind,i,:,:],Threshold)

        weightCoord = (x0.data.cpu().numpy()[0],y0.data.cpu().numpy()[0])
        distVal = distance.euclidean(scoreCoord,targetCoord)
        dist_list.append(distVal)
        coord_list.append(scoreCoord)
        target_coord_list.append(targetCoord)
        weight_coord_list.append(weightCoord)
    return dist_list,coord_list,target_coord_list,weight_coord_list

def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=3)  # b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=3)  # b,c,1,1

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=3)  # b,c,1,1

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[:, 1]  # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total

def dice_loss_norm(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=3)  #
    num = torch.sum(num, dim=0)# b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=3)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=3)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[:, 1]  # we ignore bg dice val, and take the fg
    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    return dice_total




def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def l2_normloss(input,target,size_average=True):
    criterion = torch.nn.MSELoss().cuda()  
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss

def l2_normloss_new(input,target,mask):
    loss = input - target
    loss = torch.pow(loss,2)
    loss = torch.mul(loss, mask)
    loss = loss.sum() / mask.sum()
    return loss

def l1_normloss(input,target,size_average=True):
    criterion = torch.nn.L1Loss().cuda()
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss


def l1_smooth_normloss(input,target,size_average=True):
    criterion = torch.nn.SmoothL1Loss().cuda()
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss


def l2_normloss_compete(input,target,size_average=True):
    mask = torch.sum(target, 1)
    mask = mask.expand(input.size())
    mask_ind = mask.le(0.5)
    input.masked_fill_(mask_ind, 0.0)
    mask = torch.mul(mask, 0)
    input = torch.mul(input,10)
    criterion = torch.nn.MSELoss().cuda()
    loss = criterion(input,mask)
    return loss

def l2_normloss_all(inputs,target,category_name,all_categories):
    for i in range(len(all_categories)):
        cate = all_categories[i]
        if i == 0 :
            if category_name == cate:
                loss = l2_normloss(inputs[i],target)
            else :
                loss = l2_normloss_compete(inputs[i],target)
        else:
            if category_name == cate :
                loss += l2_normloss(inputs[i],target)
            else :
                loss += l2_normloss_compete(inputs[i],target)
    return loss



def mse_loss(input, target):
    return torch.sum((input - target) ** 2)


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def write_log(log_file,target,pred_lmk,pts,epoch,batch_idx,sub_name,category_name,Threshold = 0.75):
    if not (Threshold == 0.75):
        log_file = log_file.replace('log.txt', ('log_%.2f' % Threshold))

    if batch_idx == 0 and os.path.exists(log_file):
        os.remove(log_file)

    fv = open(log_file, 'a')
    for bi in range(target.size()[0]):
        distance_list, coord_list, target_coord_list, weight_coord_list = get_distance(target, pred_lmk, bi,Threshold)
        show_str = ''
        for di in range(pts[bi].size()[0]):
            if (sum(sum(pts[0] == -1)) == 0):
                show_str = show_str + ', dist[%d]=%.4f,predlmk[%d]=(%.4f;%.4f),truelmk[%d]=(%.4f;%.4f),weightlmk[%d]=(%.4f;%.4f)' % (di,
                        distance_list[di], di,coord_list[di][1],coord_list[di][0],di, pts[bi][di, 0], pts[bi][di, 1],di,weight_coord_list[di][1],weight_coord_list[di][0])
        fv.write('epoch=%d,batch_idx=%d, subject=%s, category=%s, %s\n' % (
        epoch, batch_idx, sub_name, category_name, show_str))
    fv.close()


def save_images(results_epoch_dir,data,sub_name,cate_name,pred_lmk,target=None):
    saveOneImg(data[0, 0, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_trueGray")
    for i in range(pred_lmk.size()[1]):
        saveOneImg(pred_lmk[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_pred%d" % (i))
        if not (target is None):
            saveOneImg(target[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_true%d" % (i))


def prior_loss(input,category_name,pts,target):
    mu = {}
    std = {}
    #caculated from get_spatial_prior
    # mu['KidneyLong'] = [210.420535]
    # std['KidneyLong'] = [25.846215]
    # mu['KidneyTrans'] = [104.701820, 96.639190]
    # std['KidneyTrans'] = [17.741928, 19.972482]
    # mu['LiverLong'] = [303.206934]
    # std['LiverLong'] = [45.080338]
    # mu['SpleenLong'] = [202.573985]
    # std['SpleenLong'] = [39.253982]
    # mu['SpleenTrans'] = [190.321392, 86.738878]
    # std['SpleenTrans'] = [41.459823, 21.711744]

    pts = Variable(pts.cuda())
    # for i in input

    # grid = np.meshgrid(range(input.size()[2]), range(input.size()[3]), indexing='ij')
    x0, y0 = weighted_center(input[0, 0, :, :])
    x1, y1 = weighted_center(input[0, 1, :, :])

    dist = torch.sqrt(torch.pow(x0-x1, 2)+torch.pow(y0-y1, 2))
    truedist = torch.sqrt(torch.pow(pts[0,0,0]-pts[0,1,0], 2)+torch.pow(pts[0,0,1]-pts[0,1,1], 2))
    loss = torch.abs(dist-truedist)
    #
    if category_name == 'KidneyTrans' or category_name == 'SpleenTrans':
    #     # x2 = torch.mul(input[0, 2, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum()/input[0, 2, :, :].sum()
    #     # y2 = torch.mul(input[0, 2, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum()/input[0, 2, :, :].sum()
    #     # x3 = torch.mul(input[0, 3, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum()/input[0, 3, :, :].sum()
    #     # y3 = torch.mul(input[0, 3, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum()/input[0, 3, :, :].sum()

        # dist2 = torch.sqrt(torch.pow(x2 - x3, 2) + torch.pow(y2 - y3, 2))
        # loss += torch.abs(dist2-mu[category_name][1])
        x2, y2 = weighted_center(input[0, 2, :, :])
        x3, y3 = weighted_center(input[0, 3, :, :])
        dist = torch.sqrt(torch.pow(x2-x3, 2)+torch.pow(y2-y3, 2))
        truedist = torch.sqrt(torch.pow(pts[0,2,0]-pts[0,3,0], 2)+torch.pow(pts[0,2,1]-pts[0,3,1], 2))
        loss += torch.abs(dist-truedist)
    # # criterion = torch.nn.L1Loss().cuda()
    # # loss = criterion(dist,mu[category_name][0])

    return loss

class Trainer(object):

    def __init__(self, cuda, model, optimizer,loss_fun,
                train_loader,test_loader,lmk_num,view,
                out, max_epoch, network_num,batch_size,GAN,
                do_classification=True,do_landmarkdetect=True,
                size_average=False, interval_validate=None,
                compete = False,onlyEval=False):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.interval_validate = interval_validate
        self.network_num = network_num

        self.do_classification = do_classification
        self.do_landmarkdetect = do_landmarkdetect


        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.lmk_num = lmk_num
        self.GAN = GAN
        self.onlyEval = onlyEval
        if self.GAN:
            GAN_lr = 0.0002
            input_nc = 3
            output_nc = self.lmk_num
            ndf = 64
            norm_layer = torchsrc.models.get_norm_layer(norm_type='batch')
            gpu_ids = [0]
            self.netD = torchsrc.models.NLayerDiscriminator(input_nc+output_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=True, gpu_ids=gpu_ids)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=GAN_lr, betas=(0.5, 0.999))
            self.netD.cuda()
            self.netD.apply(torchsrc.models.weights_init)
            pool_size = 10
            self.fake_AB_pool = ImagePool(pool_size)
            no_lsgan = True
            self.Tensor = torch.cuda.FloatTensor if gpu_ids else torch.Tensor
            self.criterionGAN = torchsrc.models.GANLoss(use_lsgan=not no_lsgan, tensor=self.Tensor)


        self.max_epoch = max_epoch
        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0

        self.compete = compete
        self.batch_size = batch_size
        self.view = view
        self.loss_fun = loss_fun


    def forward_step(self, data, category_name):
        if category_name == 'KidneyLong':
            pred_lmk = self.model(data, 'KidneyLong')
        elif category_name == 'KidneyTrans':
            pred_lmk = self.model(data, 'KidneyTrans')
        elif category_name == 'LiverLong':
            pred_lmk = self.model(data, 'LiverLong')
        elif category_name == 'SpleenLong':
            pred_lmk = self.model(data, 'SpleenLong')
        elif category_name == 'SpleenTrans':
            pred_lmk = self.model(data, 'SpleenTrans')
        return pred_lmk

    def backward_D(self,real_A,real_B,fake_B):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((real_A, fake_B), 1))
        pred_fake = self.netD.forward(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = self.netD.forward(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Combined loss
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self,real_A,fake_B):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        return loss_G_GAN




    def validate(self):
        self.model.train()
        out = osp.join(self.out, 'seg_output')
        out_vis = osp.join(self.out, 'visualization')
        results_epoch_dir = osp.join(out,'epoch_%04d' % self.epoch)
        mkdir(results_epoch_dir)
        results_vis_epoch_dir = osp.join(out_vis, 'epoch_%04d' % self.epoch)
        mkdir(results_vis_epoch_dir)

        prev_sub_name = 'start'
        prev_view_name = 'start'

        for batch_idx, (data,target,target2ch,sub_name,view,img_name) in tqdm.tqdm(
                # enumerate(self.test_loader), total=len(self.test_loader),
                enumerate(self.test_loader), total=len(self.test_loader),
                desc='Valid epoch=%d' % self.epoch, ncols=80,
                leave=False):
            # if batch_idx>1000:
            #     return
            #

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data,volatile=True), Variable(target,volatile=True)

            # need_to_run = False
            # for sk in range(len(sub_name)):
            #     batch_finish_flag = os.path.join(results_epoch_dir, sub_name[sk], ('%s_%s.nii.gz' % (sub_name[sk], view[sk])))
            #     if not (os.path.exists(batch_finish_flag)):
            #         need_to_run = True
            # if not need_to_run:
            #     continue
            #
            pred = self.model(data)

            # imgs = data.data.cpu()
            lbl_pred = pred.data.max(1)[1].cpu().numpy()[:, 0, :, :]

            batch_num = lbl_pred.shape[0]
            for si in range(batch_num):
                curr_sub_name = sub_name[si]
                curr_view_name = view[si]
                curr_img_name = img_name[si]

                # out_img_dir = os.path.join(results_epoch_dir, curr_sub_name)
                # finish_flag = os.path.join(out_img_dir,('%s_%s.nii.gz'%(curr_sub_name,curr_view_name)))
                # if os.path.exists(finish_flag):
                #     prev_sub_name = 'start'
                #     prev_view_name = 'start'
                #     continue

                if prev_sub_name == 'start':
                    seg = np.zeros([512,512,512],np.uint8)
                    slice_num = 0
                elif not(prev_sub_name==curr_sub_name and prev_view_name==curr_view_name):
                    out_img_dir = os.path.join(results_epoch_dir, prev_sub_name)
                    mkdir(out_img_dir)
                    out_nii_file = os.path.join(out_img_dir,('%s_%s.nii.gz'%(prev_sub_name,prev_view_name)))
                    seg_img = nib.Nifti1Image(seg, affine=np.eye(4))
                    nib.save(seg_img, out_nii_file)
                    seg = np.zeros([512, 512, 512], np.uint8)
                    slice_num = 0

                test_slice_name = ('slice_%04d.png'%(slice_num+1))
                assert test_slice_name == curr_img_name
                seg_slice = lbl_pred[si, :, :].astype(np.uint8)
                if curr_view_name == 'view1':
                    seg[slice_num,:,:] = seg_slice
                elif curr_view_name == 'view2':
                    seg[:,slice_num,:] = seg_slice
                elif curr_view_name == 'view3':
                    seg[:, :, slice_num] = seg_slice

                slice_num+=1
                prev_sub_name = curr_sub_name
                prev_view_name = curr_view_name


        out_img_dir = os.path.join(results_epoch_dir, curr_sub_name)
        mkdir(out_img_dir)
        out_nii_file = os.path.join(out_img_dir, ('%s_%s.nii.gz' % (curr_sub_name, curr_view_name)))
        seg_img = nib.Nifti1Image(seg, affine=np.eye(4))
        nib.save(seg_img, out_nii_file)

            #     out_img_dir = os.path.join(results_epoch_dir, sub_name[si], view[si])
            #     mkdir(out_img_dir)
            #     out_mat_file = os.path.join(out_img_dir,img_name[si].replace('.png','.mat'))
            #     if not os.path.exists(out_mat_file):
            #         out_dict = {}
            #         out_dict["sub_name"] = sub_name[si]
            #         out_dict["view"] = view[si]
            #         out_dict['img_name'] = img_name[si].replace('.png','.mat')
            #         out_dict["seg"] = seg
            #         sio.savemat(out_mat_file, out_dict)

            # if not(sub_name[0] == '010-006-001'):
            #     continue
            #
            # lbl_true = target.data.cpu()
            # for img, lt, lp, name, view, fname in zip(imgs, lbl_true, lbl_pred,sub_name,view,img_name):
            #     img, lt = self.test_loader.dataset.untransform(img, lt)
            #     if lt.sum()>5000:
            #         viz = fcn.utils.visualize_segmentation(
            #             lbl_pred = lp, lbl_true = lt, img = img, n_class=2)
            #         out_img_dir = os.path.join(results_vis_epoch_dir,name,view)
            #         mkdir(out_img_dir)
            #         out_img_file = os.path.join(out_img_dir,fname)
            #         if not (os.path.exists(out_img_file)):
            #             skimage.io.imsave(out_img_file, viz)




    def train(self):
        self.model.train()
        out = osp.join(self.out, 'visualization')
        mkdir(out)
        log_file = osp.join(out, 'training_loss.txt')
        fv = open(log_file, 'a')

        for batch_idx, (data, target, target2ch, sub_name, view, img_name) in tqdm.tqdm(
            enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            #iteration = batch_idx + self.epoch * len(self.lmk_train_loader)

            # if not(sub_name[0] == '006-002-003' and view[0] =='view3' and img_name[0] == 'slice_0288.png'):
            #     continue

            if self.cuda:
                data, target, target2ch = data.cuda(), target.cuda(), target2ch.cuda()
            data, target, target2ch = Variable(data), Variable(target), Variable(target2ch)

            pred = self.model(data)
            self.optim.zero_grad()
            if self.GAN:
                self.optimizer_D.zero_grad()
                self.backward_D(data,target2ch,pred)
                self.optimizer_D.step()
                loss_G_GAN = self.backward_G(data,pred)
                if self.loss_fun == 'cross_entropy':
                    arr = np.array([1,10])
                    weight = torch.from_numpy(arr).cuda().float()
                    loss_G_L2 = cross_entropy2d(pred, target, weight=weight, size_average=True)
                elif self.loss_fun == 'Dice':
                    loss_G_L2 = dice_loss(pred,target2ch)
                elif self.loss_fun == 'Dice_norm':
                    loss_G_L2 = dice_loss_norm(pred, target2ch)
                loss = loss_G_GAN + loss_G_L2*100

                fv.write('--- epoch=%d, batch_idx=%d, D_loss=%.4f, G_loss=%.4f, L2_loss = %.4f \n' % (
                    self.epoch, batch_idx, self.loss_D.data[0], loss_G_GAN.data[0],loss_G_L2.data[0] ))

                if batch_idx%10 == 0:
                    print('--- epoch=%d, batch_idx=%d, D_loss=%.4f, G_loss=%.4f, L2_loss_loss = %.4f  \n' % (
                    self.epoch, batch_idx, self.loss_D.data[0], loss_G_GAN.data[0],loss_G_L2.data[0] ))
            else:
                if self.loss_fun == 'cross_entropy':
                    arr = np.array([1,10])
                    weight = torch.from_numpy(arr).cuda().float()
                    loss = cross_entropy2d(pred, target, weight=weight, size_average=True)
                elif self.loss_fun == 'Dice':
                    loss = dice_loss(pred,target2ch)
                elif self.loss_fun == 'Dice_norm':
                    loss = dice_loss_norm(pred, target2ch)
            loss.backward()
            self.optim.step()
            if batch_idx % 10 == 0:
                print('epoch=%d, batch_idx=%d, loss=%.4f \n'%(self.epoch,batch_idx,loss.data[0]))
                fv.write('epoch=%d, batch_idx=%d, loss=%.4f \n'%(self.epoch,batch_idx,loss.data[0]))


        fv.close()

    def train_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            out = osp.join(self.out, 'models', self.view)
            mkdir(out)

            model_pth = '%s/model_epoch_%04d.pth' % (out, epoch)
            gan_model_pth = '%s/GAN_D_epoch_%04d.pth' % (out, epoch)





            if os.path.exists(model_pth):
                self.model.load_state_dict(torch.load(model_pth))
                # if epoch == 9:
                #     self.validate()
                if self.onlyEval:
                    self.validate()
                if self.GAN and os.path.exists(gan_model_pth):
                    self.netD.load_state_dict(torch.load(gan_model_pth))
            else:
                if not self.onlyEval:
                    self.train()
                    self.validate()
                    torch.save(self.model.state_dict(), model_pth)
                    if self.GAN:
                        torch.save(self.netD.state_dict(), gan_model_pth)




