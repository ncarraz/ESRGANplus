import math
import random
import sys

#sys.path = sys.path = ["/private/home/broz/workspaces/nevergrad"] + sys.path
import argparse
import os
import itertools
from sys import platform

parser = argparse.ArgumentParser()
parser.add_argument("--opt_path", help="Path to the .yml options file", default="/content/Tarsier/codes/options/test/test_ESRGAN_set5.yml")
parser.add_argument("--penalization", help="penalization factor", default=5e-3, type=float)
parser.add_argument("--discriminator_coeff", help="penalization factor", default=1, type=float)
parser.add_argument("--gpu_id", help="GPU id", default=0, type=int)
parser.add_argument("--dim_per_layer", help="SQRT of noise dim per layer", default=5, type=int)
parser.add_argument("--budget", help="", default=1000, type=int)
parser.add_argument("--optimizer", help="", type=str, default="DiagonalCMA")
parser.add_argument("--penalisation_mode", help="", type=str, default="l2")
parser.add_argument("--log_path", help="", type=str, default="log_test/")
parser.add_argument("--images_dir", help="", type=str, default="")
parser.add_argument("--run_on_cluster", help="to run on the cluster with submitit", action="store_true", default=False)
parser.add_argument("--multiobjective", help="Enable Multiobjective optimization", action="store_true", default=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction
import os.path as osp
import logging
import time
from koniq.script_prediction import Koncept512Predictor

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import numpy as np
import torch
import submitit
import cv2
from scipy.stats import norm

loggers = {}

#dir_root = "/Users/broz/workspaces/tests_malagan/malagan/" if platform == "darwin" else "/private/home/broz/workspaces/tests_malagan/malagan/"
dir_root = "/content/Tarsier/"

class NNScorer:
    def __init__(self, opt_path_, penalisation=1.0, image_index=0, penalisation_mode='l2', discriminator_coeff=1.0,
                 images_dir="", clamp_parameter=None, blur_based_penalization=None, koncept_coeff=1.0, use_pessimistic_score=True):
        self.use_pessimistic_score = use_pessimistic_score
        self.blur_based_penalization = blur_based_penalization
        self.clamp_parameter = clamp_parameter
        self.images_dir = images_dir
        self.discriminator_coeff = discriminator_coeff
        self.koncept_coeff = koncept_coeff
        global loggers
        self.image_index = image_index
        self.opt_path = opt_path_
        self.penalisation_lambda = penalisation
        self.opt = option.parse(opt_path_, is_train=False)
        self.opt = option.dict_to_nonedict(self.opt)
        self.penalisation_mode = penalisation_mode

        self.recommendation=False
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
                                                  aux_root="/content/drive/MyDrive/ESRGAN/pretrained_models/", compute_grads=True)
        self.should_save_image = True

        self.model = create_model(self.opt)
        for i, image_data in enumerate(self.test_loader):
            if i == self.image_index:
                self.data = image_data
                break

    def get_score(self, toto, multiobjective=False, subset_method="random"):

        test_start_time = time.time()

        self.need_GT = False if self.test_loader.dataset.opt['dataroot_GT'] is None else True
        self.model.feed_data(self.data, need_GT=self.need_GT)
        img_path = self.data['GT_path'][0] if self.need_GT else self.data['LQ_path'][0]
        self.img_name = osp.splitext(osp.basename(img_path))[0]
        self.model.test(torch.from_numpy(toto).float().cuda())
        self.visuals = self.model.get_current_visuals(need_GT=self.need_GT)

        self.sr_img = util.tensor2img(self.visuals['SR'])  # uint8

        if self.optim_algo is None:
            with open("optim_method.info") as optim_method:
                lines = [line for line in optim_method]
                assert len(lines) == 1, 'There were several methods in optim_method.info'
                self.optim_algo = lines[0]

        # save images
        suffix = self.opt['suffix']
        image_name = self.img_name + "_koncept_512_" + self.optim_algo + (suffix if suffix else '') + ('_Multiobjective_' + subset_method if multiobjective else '') + '.png'
        save_img_path = osp.join(self.dataset_dir, image_name)
        if "Baseline" in self.optim_algo:
            if os.path.exists(save_img_path):
                self.logger.info("The baseline {} already exists.".format(save_img_path))
            else:
                time.sleep(random.random())
                if os.path.exists(save_img_path):
                    self.logger.info("The baseline {} already exists.".format(save_img_path))
        else:
            if self.should_save_image:
                #self.logger.info("Saving image in " + save_img_path)
                util.save_img(self.sr_img, save_img_path)

        # test_results = OrderedDict()
        # test_results['psnr'] = []
        # test_results['ssim'] = []
        # test_results['psnr_y'] = []
        # test_results['ssim_y'] = []
        # self.compute_psnr_ssim(test_results, prefix=optim_algo)
        np_image = np.moveaxis(self.model.fake_H.data.cpu().numpy(), 1, -1)
        raw_koncept_output, _ = self.quality_scorer.predict(np_image, repeats=1)

        fake_image_dim = self.model.fake_H.size()
        #print("fake image dimension:{}".format(fake_image_dim))
        dim_x, dim_y = fake_image_dim[2], fake_image_dim[3]
        nb_x = dim_x // 128
        nb_y = dim_y // 128
        discriminator_preds = []
        if self.discriminator_coeff:
            for i in range(nb_x):
                for j in range(nb_y):
                    fake_image_patch = self.model.fake_H[:, :, i * 128:(i + 1) * 128, j * 128: (j + 1) * 128]
                    with torch.no_grad():
                        discriminator_preds.append(self.model.netD(fake_image_patch).cpu().numpy()[0][0])
        else:
            discriminator_preds.append(0)
        discriminator_preds = np.array(discriminator_preds)
        if "Baseline" in self.optim_algo:
            self.baseline_koncept_output = raw_koncept_output
            self.baseline_discriminator_preds = discriminator_preds
            self.baseline_blur_score = blur_score(self.sr_img) / 1000

        raw_l2 = (np.array(toto) ** 2).mean()
        l2 = self.penalisation_lambda * raw_l2
        if self.blur_based_penalization == 'std':
            l2 *= math.sqrt(self.baseline_blur_score / 1000)
        if self.blur_based_penalization == 'variance':
            l2 *= self.baseline_blur_score / 1000

        relative_discriminator_scores = discriminator_preds - self.baseline_discriminator_preds
        #print('Discriminator scores: {}'.format(discriminator_preds))
        """self.logger.info(
            f'Mean discriminator score: {relative_discriminator_scores.mean()}, minimum discriminator score: {relative_discriminator_scores.min()}')
        """
        discriminator_score = self.pessimistic_score(relative_discriminator_scores.min())
        weighted_discriminator_score = discriminator_score * self.discriminator_coeff

        score = self.pessimistic_score(raw_koncept_output - self.baseline_koncept_output)
        """self.logger.info(
            f"raw Koncept output {raw_koncept_output}, baseline score {self.baseline_koncept_output}, score {score}")
        self.logger.info(
            f"raw l2: {raw_l2}, baseline blur {self.baseline_blur_score}, factor {self.penalisation_lambda}")
        """
        koncept_quality_score = self.koncept_coeff * score
        total_score = koncept_quality_score - l2 + weighted_discriminator_score
        """self.logger.info(
            'Koncept512Score: {:.6f}, Penalization: {}, Discriminator score: {}, total score: {},  image name {}'.format(koncept_quality_score, l2,
                                                                                                       weighted_discriminator_score,
                                                                                                       total_score,
                                                                                                       image_name) +  (' #recommendation_scores ' if self.recommendation else ''))
        """
        return (-koncept_quality_score, l2, -weighted_discriminator_score) if multiobjective else total_score

    def pessimistic_score(self, raw_score):
        if not self.use_pessimistic_score:
            return raw_score
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


def optimize_noise(arguments, multiobjective=False):
    #subset_method = "random"
    image_, penalization_lambda, budget, dim, tool, discriminator_coeff, clamping, blur_based_penalization, koncept_coeff, use_pessimistic_score = arguments
    if discriminator_coeff == 'infer20':
        discriminator_coeff = penalization_lambda / 20.0
    if discriminator_coeff == 'infer25':
        discriminator_coeff = penalization_lambda / 25.0

    #sys.path = sys.path = ["/private/home/broz/workspaces/nevergrad"] + sys.path
    import nevergrad as ng
    baseline_score = 0
    computed_baseline = False

    scorer = NNScorer(opt_path, penalisation=penalization_lambda, image_index=image_, images_dir=args.images_dir,
                      clamp_parameter=clamping, blur_based_penalization=blur_based_penalization)
    scorer.discriminator_coeff = discriminator_coeff
    scorer.koncept_coeff = koncept_coeff
    scorer.use_pessimistic_score = use_pessimistic_score

    def malagan(x):
        print("#toto parametrisation is {}".format(x))
        return -scorer.get_score(x)
    
    def moomalagan(x):
        return list(scorer.get_score(x, multiobjective=True))
    
    #moomalagan = MultiobjectiveFunction(lambda x: list(scorer.get_score(x, multiobjective=True)))

    if not computed_baseline:
        print("The baseline hasn't been computed yet. Computing it")
        scorer.optim_algo = "Baseline"
        baseline_optimizer = ng.optimizers.registry["OnePlusOne"](parametrization=dim, budget=1)
        recommendation = baseline_optimizer.minimize(malagan, verbosity=0)
        teste = malagan(recommendation.value)
        baseline_score = teste
        print("The baseline score is {}".format(baseline_score))
        computed_baseline = True
    scorer.optim_algo = "{}_penal{}_{}_disc{}_clamping{}_bud{}_blur_{}".format(tool, scorer.penalisation_lambda, dim,
                                                                       scorer.discriminator_coeff, clamping, budget, blur_based_penalization)
    if not use_pessimistic_score:
        scorer.optim_algo += '_raw_scores'
    optimizer = ng.optimizers.registry[tool](parametrization=dim, budget=budget, num_workers=2)
    recommendation = optimizer.minimize(malagan if not multiobjective else moomalagan, verbosity=0)
    print("#recommendation {}".format(recommendation))  # optimal args and kwargs
    scorer.recommendation = True
    if multiobjective:
        for subset_method in ["random", "loss-covering", "hypervolume", "EPS"]:
            for i, x in enumerate(optimizer.pareto_front(9, subset=subset_method, subset_tentatives=500)):
                teste = malagan(x.value) 
                #scorer.logger.info("Budget: {}, Tool: {}, Value : {}, Baseline improvement: {}, Optim algo: {}".format(budget, tool, teste, teste - baseline_score, scorer.optim_algo))

                scorer.need_GT = False if scorer.test_loader.dataset.opt['dataroot_GT'] is None else True
                scorer.model.feed_data(scorer.data, need_GT=scorer.need_GT)
                img_path = scorer.data['GT_path'][0] if scorer.need_GT else scorer.data['LQ_path'][0]
                scorer.img_name = osp.splitext(osp.basename(img_path))[0]
                scorer.model.test(torch.from_numpy(x.value).float().cuda())
                scorer.visuals = scorer.model.get_current_visuals(need_GT=scorer.need_GT)
                scorer.sr_img = util.tensor2img(scorer.visuals['SR'])  # uint8

                suffix = scorer.opt['suffix']
                image_name = scorer.img_name + "_koncept_512_" + scorer.optim_algo + (suffix if suffix else '') + ('_Multiobjective_' + subset_method if multiobjective else '') + '_' + str(i) + '_'+ '.png'
                save_img_path = osp.join("/content/drive/MyDrive/MOO images", image_name)

                scorer.logger.info("Saving image in {}".format(save_img_path))
                util.save_img(scorer.sr_img, save_img_path)
        
        
    else:
        teste = malagan(recommendation.value)
        print(budget, tool, "Score:", teste, teste - baseline_score, scorer.optim_algo, "#results")


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
    assert args.opt_path == "/content/Tarsier/codes/options/test/test_ESRGAN_set5.yml", "option is not recognized"
    num_images = 18
    
    
if __name__ == '__main__':
    log_path = os.path.join(args.log_path, "%j")
    executor = submitit.AutoExecutor(folder=log_path)

    partition = "learnfair" #'learnfair' #"uninterrupted"  # dev
    executor.update_parameters(gpus_per_node=1, cpus_per_task=3, timeout_min=2500, partition=partition, comment="srgan")

    image_indices = range(num_images)
    penalizations = [1.0]
    budgets = [10000]  # [1000, 10000]
    dims = [dim_per_layer ** 2 * 23 * 3 for dim_per_layer in [20]]
    tools = ["DE"] #["CMA", "DiagonalCMA", 'OnePlusOne', 'PSO', 'DE', 'TBPSA']
    discriminator_coeffs = [1.0]
    koncept_coeffs = [1.0]
    clamping = [None]
    blur_based_penalizations = [None]
    use_pessimistic_scores = [True]

    args_array = itertools.product(image_indices, penalizations, budgets, dims, tools, discriminator_coeffs, clamping,
                                   blur_based_penalizations, koncept_coeffs, use_pessimistic_scores)
    # map = executor.map_array(optimize_noise, args_array)
    # for argument in args_array:
    #     optimize_noise(argument)
    run_on_cluster = args.run_on_cluster
    moo = args.multiobjective
    
    if run_on_cluster:
        executor = submitit.AutoExecutor(folder=log_path)
        executor.update_parameters(gpus_per_node=1, cpus_per_task=3, timeout_min=2500, partition=partition,
                                   comment="srgan")
        executor.update_parameters(array_parallelism=400)
        map = executor.map_array(optimize_noise, args_array)
        print(map)
    else:
        for argument in args_array:
            optimize_noise(argument, moo)
