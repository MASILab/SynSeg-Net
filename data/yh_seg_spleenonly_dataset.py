import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import random_crop_yh

class yhSegDatasetSpleenOnly(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.dir_A = opt.raw_MRI_dir
        self.dir_B = opt.raw_CT_dir
        self.dir_Seg = opt.raw_MRI_seg_dir

        self.A_paths = opt.imglist_MRI
        self.B_paths = opt.imglist_CT

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
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
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]
        Seg_path = A_path.replace(self.dir_A,self.dir_Seg)
        Seg_path = Seg_path.replace('_rawimg','_organlabel')

        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('L')
        Seg_img = Image.open(Seg_path).convert('I')
        B_img = Image.open(B_path).convert('L')

        A_img = self.transforms_scale(A_img)
        B_img = self.transforms_scale(B_img)
        Seg_img = self.transforms_seg_scale(Seg_img)

        if not self.skipcrop:
            [A_img,Seg_img] = self.transforms_crop([A_img, Seg_img])
            [B_img] = self.transforms_crop([B_img])

        A_img = self.transforms_toTensor(A_img)
        B_img = self.transforms_toTensor(B_img)
        Seg_img = self.transforms_toTensor(Seg_img)

        A_img = self.transforms_normalize(A_img)
        B_img = self.transforms_normalize(B_img)

        Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.fineSize, self.opt.fineSize)
        Seg_imgs[0, :, :] = Seg_img == 0
        Seg_imgs[1, :, :] = Seg_img == 1


        return {'A': A_img, 'B': B_img, 'Seg': Seg_imgs, 'Seg_one': Seg_img,
                'A_paths': A_path, 'B_paths': B_path, 'Seg_paths':Seg_path}


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
