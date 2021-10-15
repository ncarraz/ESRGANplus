import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import koniq
from koniq.script_prediction import Koncept512Predictor

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import tensorflow as tf

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

test_loader = test_loaders[0]
model = create_model(opt)

test_set_name = test_loader.dataset.opt['name']
logger.info('\nTesting [{:s}]...'.format(test_set_name))
test_start_time = time.time()
dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
util.mkdir(dataset_dir)

# test_results = OrderedDict()
# test_results['psnr'] = []
# test_results['ssim'] = []
# test_results['psnr_y'] = []
# test_results['ssim_y'] = []


data_index = 0
for data in test_loader:
    data_index += 1
    if data_index > 200:
        break
    need_GT = False # if test_loader.dataset.opt['dataroot_GT'] is None else True
    model.feed_data(data, need_GT=need_GT)
    img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
    img_name = osp.splitext(osp.basename(img_path))[0]

    model.test()
    visuals = model.get_current_visuals(need_GT=need_GT)

    sr_img = util.tensor2img(visuals['SR'])  # uint8

    with open("optim_method.info") as optim_method:
        lines = [line for line in optim_method]
        assert len(lines) == 1, 'There were several methods in optim_method.info'
        optim_algo = lines[0]

    # save images
    suffix = opt['suffix']
    image_name = img_name + "_koncept_512_" + optim_algo + (suffix if suffix else '') + '.png'
    save_img_path = osp.join(dataset_dir, image_name)
    logger.info("Saving image in " + save_img_path)
    util.save_img(sr_img, save_img_path)

    score = Koncept512Predictor(dataset_dir, aux_root="/private/home/broz/workspaces/tests_malagan/malagan/codes/koniq/").predict(image_name, repeats=100)
    logger.info('Concept512Score:{:.6f}'.format(score))
    break
