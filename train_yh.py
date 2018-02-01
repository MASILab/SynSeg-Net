import time
import os
import sublist
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

opt = TrainOptions().parse()



# Method = 'ImageOnly'
Method = opt.yh_data_model


raw_MRI_dir = '/home-local/Cycle_Deep/Data2D_bothimgandseg_andmask/MRI/img'
raw_MRI_seg_dir = '/home-local/Cycle_Deep/Data2D_bothimgandseg_andmask/MRI/seg'
raw_CT_dir = '/home-local/Cycle_Deep/Data2D_bothimgandseg_andmask/CT/img'
sub_list_dir = './sublist'


TrainOrTest = opt.yh_run_model #'Train' #


#evaluation
if TrainOrTest == 'Test':

    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.isTrain = False
    opt.phase = 'test'
    opt.no_dropout = True

    cycle_output_dir = opt.test_seg_output_dir
    # if not os.path.exists(cycle_output_dir):
    #     cycle_output_dir = '/scratch/huoy1/projects/DeepLearning/Cycle_Deep/Output/CycleTest'


    mkdir(sub_list_dir)
    sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri.txt')
    sub_list_CT = os.path.join(sub_list_dir, 'sublist_CT.txt')

    imglist_MRI = sublist.dir2list(raw_MRI_dir, sub_list_MRI)
    imglist_CT = sublist.dir2list(raw_CT_dir, sub_list_CT)

    imglist_MRI, imglist_CT = sublist.equal_length_two_list(imglist_MRI, imglist_CT);

    # input the opt that we want
    opt.raw_MRI_dir = raw_MRI_dir
    opt.raw_MRI_seg_dir = raw_MRI_seg_dir
    opt.raw_CT_dir = raw_CT_dir
    opt.imglist_MRI = imglist_MRI
    opt.imglist_CT = imglist_CT

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)


    print('#testing images = %d' % dataset_size)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images_to_dir(cycle_output_dir, visuals, img_path)

elif TrainOrTest == 'TestSeg':
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.isTrain = False
    opt.phase = 'test'
    opt.no_dropout = True
    seg_output_dir = opt.test_seg_output_dir

    opt.test_CT_dir = opt.test_CT_dir

    if opt.custom_sub_dir == 1:
        sub_list_dir = os.path.join(seg_output_dir,'sublist')
        mkdir(sub_list_dir)
    test_img_list_file = os.path.join(sub_list_dir,'test_CT_list.txt')
    opt.imglist_testCT = sublist.dir2list(opt.test_CT_dir, test_img_list_file)
    opt.imglist_testCT = sublist.dir2list(opt.test_CT_dir, test_img_list_file)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_seg_images_to_dir(seg_output_dir, visuals, img_path)


elif TrainOrTest == 'Train':

    mkdir(sub_list_dir)
    sub_list_MRI = os.path.join(sub_list_dir, 'sublist_mri.txt')
    sub_list_CT = os.path.join(sub_list_dir, 'sublist_CT.txt')

    imglist_MRI = sublist.dir2list(raw_MRI_dir, sub_list_MRI)
    imglist_CT = sublist.dir2list(raw_CT_dir, sub_list_CT)

    imglist_MRI, imglist_CT = sublist.equal_length_two_list(imglist_MRI, imglist_CT);

    # input the opt that we want
    opt.raw_MRI_dir = raw_MRI_dir
    opt.raw_MRI_seg_dir = raw_MRI_seg_dir
    opt.raw_CT_dir = raw_CT_dir
    opt.imglist_MRI = imglist_MRI
    opt.imglist_CT = imglist_CT
    opt.crossentropy_weight = [1,1,opt.weight_2,opt.weight_3,1,opt.weight_7,1]

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)



    print('#training images = %d' % dataset_size)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    print('#model created')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()
