import math
import sys

sys.path = sys.path = ["/private/home/broz/workspaces/nevergrad"] + sys.path
import argparse
import os
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--opt_path", help="Path to the .yml options file", default="options/test/test_ESRGAN_set5.yml")
parser.add_argument("--penalization", help="penalization factor", default=5e-3, type=float)
parser.add_argument("--discriminator_coeff", help="penalization factor", default=0, type=float)
parser.add_argument("--gpu_id", help="GPU id", default=0, type=int)
parser.add_argument("--dim_per_layer", help="SQRT of noise dim per layer", default=5, type=int)
parser.add_argument("--budget", help="", default=1000, type=int)
parser.add_argument("--optimizer", help="", type=str, default="DiagonalCMA")
parser.add_argument("--penalisation_mode", help="", type=str, default="l2")
parser.add_argument("--log_path", help="", type=str, default="log_test/")
parser.add_argument("--images_dir", help="", type=str, default="")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
import nevergrad as ng
import os.path as osp
import logging
import time
from koniq.script_prediction import Koncept512Predictor

import options.options as option
import utils.util as util
from data.util import read_img
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import numpy as np
import torch
import submitit
import cv2

sys.path = sys.path = ["/private/home/broz/workspaces/nevergrad"] + sys.path
import nevergrad as ng

from scipy.stats import norm

loggers = {}


class NNScorer:
    def __init__(self, opt_path_, images_list, penalisation=1.0, penalisation_mode='l2', discriminator_coeff=1.0,
                  clamp_parameter=None, blur_based_penalization=None):
        self.blur_based_penalization = blur_based_penalization
        self.clamp_parameter = clamp_parameter
        self.images_list = images_list
        self.discriminator_coeff = discriminator_coeff
        global loggers
        self.opt_path = opt_path_
        self.penalisation_lambda = penalisation
        self.opt = option.parse(opt_path_, is_train=False)
        self.opt = option.dict_to_nonedict(self.opt)
        self.penalisation_mode = penalisation_mode
        util.mkdirs(
            (path for key, path in self.opt['path'].items()
             if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
        self.optim_algo = None

        if loggers.get('base'):
            self.logger = loggers.get('base')
        else:
            util.setup_logger('base', self.opt['path']['log'], 'test_' + self.opt['name'], level=logging.INFO,
                              screen=True,
                              tofile=True)
            # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
            self.logger = logging.getLogger('base')
            self.logger.info(option.dict2str(self.opt))
            loggers['base'] = self.logger

        # Create test dataset and dataloader
        test_loaders = []
        for phase, dataset_opt in sorted(self.opt['datasets'].items()):
            test_set = create_dataset(dataset_opt)
            test_loader = create_dataloader(test_set, dataset_opt)
            self.logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
            test_loaders.append(test_loader)

        self.test_loader = test_loaders[0]
        test_set_name = self.test_loader.dataset.opt['name']
        self.logger.info('\nTesting [{:s}]...'.format(test_set_name))
        self.dataset_dir = osp.join(self.opt['path']['results_root'], test_set_name, self.images_dir)
        util.mkdir(self.dataset_dir)
        self.quality_scorer = Koncept512Predictor(self.dataset_dir,
                                                  aux_root="/private/home/broz/workspaces/tests_malagan/malagan/codes/koniq/")

        self.model = create_model(self.opt)
        self.baseline_koncept_output = {}
        self.baseline_discriminator_preds = {}
        self.baseline_blur_score = {}

    def get_score(self, toto):
        results = {}
        test_start_time = time.time()
        for img_path in [img for img in self.images_list if '_Baseline.png' in img]:
            self.get_score_for_image(img_path, toto, baseline=True)

        for i, img_path in enumerate(self.images_list):
            results[img_path] = self.get_score_for_image(img_path, toto)

        return results

    def get_score_for_image(self, img_path, toto, baseline=False):
        data = read_img(None, path=img_path)
        if data.shape[2] == 3:
            data = data[:, :, [2, 1, 0]]
        data = torch.from_numpy(np.ascontiguousarray(np.transpose(data, (2, 0, 1)))).float()
        np_image = np.moveaxis(data.numpy(), 1, -1)
        raw_koncept_output, _ = self.quality_scorer.predict(np_image, repeats=1)
        generated_image = data
        fake_image_dim = generated_image.size()
        print("fake image dimension:{}".format(fake_image_dim))
        dim_x, dim_y = fake_image_dim[2], fake_image_dim[3]
        nb_x = dim_x // 128
        nb_y = dim_y // 128
        discriminator_preds = []
        if self.discriminator_coeff:
            for i in range(nb_x):
                for j in range(nb_y):
                    fake_image_patch = generated_image[:, :, i * 128:(i + 1) * 128, j * 128: (j + 1) * 128]
                    with torch.no_grad():
                        discriminator_preds.append(self.model.netD(fake_image_patch).cpu().numpy()[0][0])
        else:
            discriminator_preds.append(0)
        discriminator_preds = np.array(discriminator_preds)
        image_identifier = img_path.split('/')[-1].split('_')[0]
        if baseline:
            self.baseline_koncept_output[image_identifier] = raw_koncept_output
            self.baseline_discriminator_preds[image_identifier] = discriminator_preds
            self.baseline_blur_score[image_identifier] = blur_score(self.sr_img) / 1000
        assert( image_identifier in self.baseline_koncept_output, {f"identifier {image_identifier} not in baseline output"})
        raw_l2 = (np.array(toto) ** 2).mean()
        l2 = self.penalisation_lambda * raw_l2
        if self.blur_based_penalization == 'std':
            l2 *= math.sqrt(self.baseline_blur_score[image_identifier] / 1000)
        if self.blur_based_penalization == 'variance':
            l2 *= self.baseline_blur_score[image_identifier] / 1000
        relative_discriminator_scores = discriminator_preds - self.baseline_discriminator_preds[image_identifier]
        print('Discriminator scores: {}'.format(discriminator_preds))
        self.logger.info(
            f'Mean discriminator score: {relative_discriminator_scores.mean()}, minimum discriminator score: {relative_discriminator_scores.min()}')
        discriminator_score = self.pessimistic_score(relative_discriminator_scores.min())
        weighted_discriminator_score = discriminator_score * self.discriminator_coeff
        score = self.pessimistic_score(raw_koncept_output - self.baseline_koncept_output[image_identifier])
        self.logger.info(
            f"raw Koncept output {raw_koncept_output}, baseline score {self.baseline_koncept_output[image_identifier]}, score {score}")
        self.logger.info(
            f"raw l2: {raw_l2}, baseline blur {self.baseline_blur_score[image_identifier]}, factor {self.penalisation_lambda}")
        self.logger.info(
            'Koncept512Score:{:.6f}, Penalization: {}, Discriminator score: {},  image name {}'.format(score, l2,
                                                                                                       weighted_discriminator_score,
                                                                                                       img_path))
        return score - l2 + weighted_discriminator_score

    def pessimistic_score(self, raw_score):
        if raw_score > 0:
            score = math.log(1 + raw_score)
            if self.clamp_parameter:
                score = min(score, self.clamp_parameter)
            return score
        else:
            return raw_score

    # def compute_psnr_ssim(self, test_results, prefix):
    #     if self.need_GT:
    #         gt_img = util.tensor2img(self.visuals['GT'])
    #         gt_img = gt_img / 255.
    #         self.sr_img = self.sr_img / 255.
    #
    #         crop_border = self.opt['crop_border'] if self.opt['crop_border'] else self.opt['scale']
    #         if crop_border == 0:
    #             cropped_sr_img = self.sr_img
    #             cropped_gt_img = gt_img
    #         else:
    #             cropped_sr_img = self.sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
    #             cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]
    #
    #         psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
    #         ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
    #         test_results['psnr'].append(psnr)
    #         test_results['ssim'].append(ssim)
    #
    #         if gt_img.shape[2] == 3:  # RGB image
    #             sr_img_y = bgr2ycbcr(self.sr_img, only_y=True)
    #             gt_img_y = bgr2ycbcr(gt_img, only_y=True)
    #             if crop_border == 0:
    #                 cropped_sr_img_y = sr_img_y
    #                 cropped_gt_img_y = gt_img_y
    #             else:
    #                 cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
    #                 cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
    #             psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
    #             ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
    #             test_results['psnr_y'].append(psnr_y)
    #             test_results['ssim_y'].append(ssim_y)
    #             self.logger.info(
    #                 prefix + '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
    #                 format(self.img_name, psnr, ssim, psnr_y, ssim_y))
    #         else:
    #             self.logger.info(prefix + '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(self.img_name, psnr, ssim))
    #     else:
    #         self.logger.info(prefix + self.img_name)


def blur_score(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        print("Couldn't process image")
        exit(0)
    return variance_of_laplacian(gray)


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


dim_per_layer = args.dim_per_layer ** 2

dim = dim_per_layer * 23 * 3
opt_path = args.opt_path




if args.opt_path == "options/test/test_ESRGAN_set5.yml":
    num_images = 5
elif args.opt_path == "options/test/test_ESRGAN_set14.yml":
    num_images = 14
elif 'PIRM' in args.opt_path:
    num_images = 100
elif 'Urban' in args.opt_path:
    num_images = 100
elif 'OST' in args.opt_path:
    num_images = 300
else:
    assert args.opt_path == "options/test/test_ESRGAN.yml", "option is not recognized"
    num_images = 18
if __name__ == '__main__':
    log_path = os.path.join(args.log_path, "%j")
    executor = submitit.AutoExecutor(folder=log_path)

    partition = "uninterrupted"  # dev
    executor.update_parameters(gpus_per_node=1, cpus_per_task=3, timeout_min=2500, partition=partition, comment="srgan")

    image_indices = range(num_images)
    penalization = 1.0
    dims = [dim_per_layer ** 2 * 23 * 3 for dim_per_layer in [20]]
    tools = ["DiagonalCMA"]
    discriminator_coeff = 1.0
    clamping = None
    executor.update_parameters(array_parallelism=150)
    blur_based_penalization = None

    # args_array = itertools.product(image_indices, penalizations, budgets, dims, tools, discriminator_coeffs, clamping,
    #                                blur_based_penalizations)
    # map = executor.map_array(optimize_noise, args_array)
    baseline_score = 0
    computed_baseline = False
    images_list =
    scorer = NNScorer(opt_path, images_list=images_list, penalisation=penalization, clamp_parameter=clamping, blur_based_penalization=blur_based_penalization)
    scorer.discriminator_coeff = discriminator_coeff
    result = scorer.get_score()

    # for argument in args_array:
    #     optimize_noise(argument)
    print(map)
