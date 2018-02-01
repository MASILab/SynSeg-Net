import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import random_crop_yh

class yhTestSegDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = opt.test_CT_dir
        # self.dir_Seg = opt.test_CT_seg_dir

        self.A_paths = opt.imglist_testCT

        self.A_size = len(self.A_paths)

        if not self.opt.isTrain:
            self.skipcrop = True
        else:
            self.skipcrop = False
        # self.transform = get_transform(opt)

        if self.skipcrop:
            osize = [opt.fineSize, opt.fineSize]
        else:
            osize = [opt.loadSize, opt.loadSize]
        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.NEAREST))
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_crop_yh.randomcrop_yh(opt.fineSize))
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5)))
        self.transforms_normalize = transforms.Compose(transform_list)


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        # Seg_path = A_path.replace(self.dir_A,self.dir_Seg)
        # Seg_path = Seg_path.replace('_rawimg','_organlabel')

        A_img = Image.open(A_path).convert('L')
        # Seg_img = Image.open(Seg_path).convert('I')

        A_img = self.transforms_scale(A_img)
        # Seg_img = self.transforms_seg_scale(Seg_img)

        A_img = self.transforms_toTensor(A_img)
        # Seg_img = self.transforms_toTensor(Seg_img)

        A_img = self.transforms_normalize(A_img)

        #strategy 1
        # Seg_img[Seg_img == 6] = 4
        # Seg_img[Seg_img == 7] = 5
        # Seg_img[Seg_img == 14] = 6
        #
        # Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.fineSize, self.opt.fineSize)
        # Seg_imgs[0, :, :] = Seg_img == 0
        # Seg_imgs[1, :, :] = Seg_img == 1
        # Seg_imgs[2, :, :] = Seg_img == 2
        # Seg_imgs[3, :, :] = Seg_img == 3
        # Seg_imgs[4, :, :] = Seg_img == 4
        # Seg_imgs[5, :, :] = Seg_img == 5
        # Seg_imgs[6, :, :] = Seg_img == 6

        #strategy 2
        # Seg_img[Seg_img == 2] = 3
        # Seg_img[Seg_img == 14] = 3
        # Seg_img[Seg_img == 3] = 3
        # Seg_img[Seg_img == 4] = 3
        # Seg_img[Seg_img == 5] = 3
        # Seg_img[Seg_img == 7] = 3
        # Seg_img[Seg_img == 6] = 2
        #
        # Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.fineSize, self.opt.fineSize)
        # Seg_imgs[0, :, :] = Seg_img == 0
        # Seg_imgs[1, :, :] = Seg_img == 1
        # Seg_imgs[2, :, :] = Seg_img == 2
        # Seg_imgs[3, :, :] = Seg_img == 3
        Seg_imgs = 0
        Seg_path = ''

        return {'A': A_img, 'Seg': Seg_imgs,
                'A_paths': A_path, 'Seg_paths':Seg_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'TestCTDataset'
