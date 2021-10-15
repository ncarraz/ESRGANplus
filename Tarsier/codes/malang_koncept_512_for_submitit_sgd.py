import math
import sys
import torch.nn as nn

from LBFGS import LBFGS

sys.path = sys.path = ["/private/home/broz/workspaces/nevergrad"] + sys.path
import argparse
import os
import itertools
from sys import platform

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
parser.add_argument("--run_on_cluster", help="to run on the cluster with submitit", action="store_true", default=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
import nevergrad as ng
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
from torch.autograd import Variable
import matplotlib.pyplot as plt

from scipy.stats import norm
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/sgd_experiment/')
loggers = {}

dir_root = "/Users/broz/workspaces/tests_malagan/malagan/" if platform == "darwin" else "/private/home/broz/workspaces/tests_malagan/malagan/"

import functools, traceback
def gpu_mem_restore(func):
    "Reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted"
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            type, val, tb = sys.exc_info()
            traceback.clear_frames(tb)
            raise type(val).with_traceback(tb) from None
    return wrapper

class NNScorer:
    def __init__(self, opt_path_, penalisation=1.0, image_index=0, penalisation_mode='l2', discriminator_coeff=0,
                 images_dir="", clamp_parameter=None, blur_based_penalization=None):
        self.blur_based_penalization = blur_based_penalization
        self.clamp_parameter = clamp_parameter
        self.images_dir = images_dir
        self.discriminator_coeff = discriminator_coeff
        global loggers
        self.image_index = image_index
        self.opt_path = opt_path_
        self.penalisation_lambda = penalisation
        self.opt = option.parse(opt_path_, is_train=False)
        self.opt = option.dict_to_nonedict(self.opt)
        self.penalisation_mode = penalisation_mode
        self.recommendation=False
        self.baseline_discriminator_preds = 0
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
                                                  aux_root=dir_root + "codes/koniq/", compute_grads=True)

        self.model = create_model(self.opt)
        if platform == 'darwin':
            self.model.device = 'cpu'
        self.koncept_gradients_image = None
        self.total_gradient = None
        self.should_save_image = False
        for i, image_data in enumerate(self.test_loader):
            if i == self.image_index:
                self.data = image_data
                break

    @gpu_mem_restore
    def get_score(self, optim_z):

        test_start_time = time.time()
        # img_path = self.data['LQ_path'][0]
        # if osp.splitext(osp.basename(img_path))[0] != '10':
        #     continue

        self.need_GT = False if self.test_loader.dataset.opt['dataroot_GT'] is None else True
        self.model.feed_data(self.data, need_GT=self.need_GT)
        img_path = self.data['GT_path'][0] if self.need_GT else self.data['LQ_path'][0]
        self.img_name = osp.splitext(osp.basename(img_path))[0]
        new_optimized_z = Variable(torch.from_numpy(optim_z).float(), requires_grad=True)
        new_optimized_z.requires_grad = True
        for param in self.model.netG.parameters():
            param.requires_grad = False
        for param in self.model.netD.parameters():
            param.requires_grad = False
        new_z_cuda = new_optimized_z.cuda()
        new_z_cuda.retain_grad()
        # for p in self.model.netG.parameters():
        #     p.requires_grad = True
        self.model.test(new_z_cuda, with_grads=True)

        if self.optim_algo is None:
            with open("optim_method.info") as optim_method:
                lines = [line for line in optim_method]
                assert len(lines) == 1, 'There were several methods in optim_method.info'
                self.optim_algo = lines[0]

        # save images
        self.visuals = self.model.get_current_visuals(need_GT=self.need_GT)
        self.sr_img = util.tensor2img(self.visuals['SR'])  # uint8
        suffix = self.opt['suffix']
        image_name = self.img_name + "_koncept_512_" + self.optim_algo + (suffix if suffix else '') + '.png'
        save_img_path = osp.join(self.dataset_dir, image_name)
        if "Baseline" in self.optim_algo and os.path.exists(save_img_path):
            self.logger.info("The baseline {} already exists.".format(save_img_path))
        else:
            if self.should_save_image:
                self.logger.info("Saving image in " + save_img_path)
                util.save_img(self.sr_img, save_img_path)

        # test_results = OrderedDict()
        # test_results['psnr'] = []
        # test_results['ssim'] = []
        # test_results['psnr_y'] = []
        # test_results['ssim_y'] = []
        # self.compute_psnr_ssim(test_results, prefix=optim_algo)
        np_image = np.moveaxis(self.model.fake_H.data.cpu().numpy(), 1, -1)

        raw_koncept_output, koncept_gradient = self.quality_scorer.predict(np_image, repeats=1)

        self.koncept_gradients_image = koncept_gradient
        self.koncept_gradients_image = np.moveaxis(self.koncept_gradients_image, -1, 1)
        self.logger.info("Koncept gradients for fake_H:{}".format(self.koncept_gradients_image))

        # np.save('/private/home/broz/workspaces/tests_malagan/malagan/codes/koncept_gradients', self.koncept_gradients)
        print(f"koncept_gradients shape {self.koncept_gradients_image.shape}, fake_h shape {self.model.fake_H.shape}")
        fake_H_sum = (torch.from_numpy(self.koncept_gradients_image).float().to(self.model.device) * self.model.fake_H).sum()
        fake_H_sum.backward()
        gradients_koncept_score = new_z_cuda.grad.data.detach().cpu().numpy()
        self.logger.info(f"gradients koncept score: {gradients_koncept_score}")
        self.model.netG.zero_grad()
        self.model.netD.zero_grad()
        new_z_cuda.grad.zero_()

        self.model.test(new_z_cuda, with_grads=True)
        fake_image_dim = self.model.fake_H.size()
        print("fake image dimension:{}".format(fake_image_dim))
        dim_x, dim_y = fake_image_dim[2], fake_image_dim[3]
        nb_x = dim_x // 128
        nb_y = dim_y // 128

        # discriminator_preds = self.model.netD(self.model.fake_H[:, :, : 128, : 128])
        # min_discriminator_pred = discriminator_preds.mean()
        discriminator_preds = []
        if self.discriminator_coeff:
            for i in range(nb_x):
                for j in range(nb_y):
                    discriminator_preds.append(self.model.netD(self.model.fake_H[:, :, i * 128:(i + 1) * 128,
                                                               j * 128: (j + 1) * 128]))  # .cpu().numpy()[0][0])
        else:
            discriminator_preds.append(0)
        print(f"discriminator preds{discriminator_preds}")
        discriminator_preds = torch.cat(discriminator_preds)

        relative_discriminator_scores = discriminator_preds - self.baseline_discriminator_preds
        print('Discriminator scores: {}'.format(discriminator_preds))
        min_discriminator_pred = relative_discriminator_scores.mean()
        # min_discriminator_pred.requires_grad = True
        self.logger.info(
            f'Mean discriminator score: {discriminator_preds.mean()}, minimum discriminator score: {min_discriminator_pred}')
        new_z_cuda.retain_grad()

        min_discriminator_pred.backward()
        discriminator_gradients = new_z_cuda.grad.data.detach().cpu().numpy()
        self.model.netD.zero_grad()
        self.model.netG.zero_grad()
        new_z_cuda.grad.zero_()
        self.logger.info(f"discriminator gradient: {discriminator_gradients}")

        if "Baseline" in self.optim_algo:
            self.baseline_discriminator_preds = discriminator_preds.detach()
            self.baseline_blur_score = blur_score(self.sr_img) / 1000
            self.baseline_koncept_output = raw_koncept_output

        raw_l2 = (np.array(optim_z) ** 2).mean()
        new_lambda = self.penalisation_lambda
        if self.blur_based_penalization == 'std':
            new_lambda *= math.sqrt(self.baseline_blur_score / 1000)
        if self.blur_based_penalization == 'variance':
            new_lambda *= self.baseline_blur_score / 1000
        l2 = new_lambda * raw_l2
        l2_gradients = 2 * new_lambda * optim_z / len(optim_z)

        discriminator_score, discriminator_gradients = self.pessimistic_score(
            min_discriminator_pred.detach().cpu().numpy(), discriminator_gradients)
        print(f"debug {min_discriminator_pred}")
        weighted_discriminator_score = self.discriminator_coeff * discriminator_score
        discriminator_gradients *= self.discriminator_coeff
        score, gradients_koncept_score = self.pessimistic_score(raw_koncept_output - self.baseline_koncept_output, gradients_koncept_score)

        self.total_gradient = gradients_koncept_score + discriminator_gradients[0] - l2_gradients
        self.logger.info(f"Total gradient: {self.total_gradient}")

        self.logger.info(
            f"raw Koncept output {raw_koncept_output}, baseline score {self.baseline_koncept_output}, score {score}")
        self.logger.info(
            f"raw l2: {raw_l2}, baseline blur {self.baseline_blur_score}, factor {self.penalisation_lambda}")
        total_score = score - l2 + weighted_discriminator_score
        self.logger.info(
            'Koncept512Score: {:.6f}, Penalization: {}, Discriminator score: {}, total score: {},  image name {}'.format(score, l2,
                                                                                                       weighted_discriminator_score,
                                                                                                       total_score,
                                                                                                       image_name) +  (' #recommendation_scores ' if self.recommendation else ''))
        return total_score

    def pessimistic_score(self, raw_score, gradient=None):
        if raw_score > 0:
            score = math.log(1 + raw_score)
            if self.clamp_parameter:
                score = min(score, self.clamp_parameter)
            if gradient is not None:
                return score, gradient / (1 + raw_score)
            else:
                return score
        else:
            if gradient is not None:
                return raw_score, gradient
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


def optimize_noise(arguments):
    image_, penalization_lambda, budget, dim, tool, discriminator_coeff, clamping, blur_based_penalization = arguments
    if discriminator_coeff == 'infer20':
        discriminator_coeff = penalization_lambda / 20.0
    if discriminator_coeff == 'infer25':
        discriminator_coeff = penalization_lambda / 25.0

    sys.path = sys.path = ["/private/home/broz/workspaces/nevergrad"] + sys.path
    import nevergrad as ng
    baseline_score = 0
    computed_baseline = False

    scorer = NNScorer(opt_path, penalisation=penalization_lambda, image_index=image_, images_dir=args.images_dir,
                      clamp_parameter=clamping, blur_based_penalization=blur_based_penalization)
    scorer.discriminator_coeff = discriminator_coeff
    scorer.should_save_image = True

    def malagan(x):
        print("#toto parametrisation is {}".format(x))
        return -scorer.get_score(x)

    if not computed_baseline:
        print("The baseline hasn't been computed yet. Computing it")
        scorer.optim_algo = "Baseline"
        scorer.should_save_image = True
        baseline_optimizer = ng.optimizers.registry["OnePlusOne"](instrumentation=dim, budget=1)

        recommendation = baseline_optimizer.optimize(malagan)
        teste = malagan(recommendation.data)
        baseline_score = teste
        print("The baseline score is {}".format(baseline_score))
        computed_baseline = True
        scorer.should_save_image = False

    scorer.optim_algo = "{}_penal{}_{}_disc{}_clamping{}_bud{}_blur_{}".format(tool, scorer.penalisation_lambda, dim,
                                                                               scorer.discriminator_coeff, clamping,
                                                                               budget, blur_based_penalization)

    learning_rate = 0.1
    if tool == "SGD":
        optimizer = SGD(learning_rate, budget=budget, dimension=dim)
        recommendation = optimizer.optimize(scorer)
    elif tool == "Adam":
        optimizer = Adam(learning_rate, budget=budget, dimension=dim)
        recommendation = optimizer.optimize(scorer)
    elif tool == "LBFGS":
        optimizer = LBFGS(scorer=scorer, max_iter=budget, dimension=dim)
        recommendation = optimizer.step(malagan)
    else:
        optimizer = ng.optimizers.registry[tool](instrumentation=dim, budget=budget)
        recommendation = optimizer.optimize(malagan).data
    print("#recommendation {}".format(recommendation))  # optimal args and kwargs
    scorer.recommendation = True

    scorer.should_save_image = True
    teste = malagan(recommendation)
    loggers['base'].info((budget, tool, "Score:", teste, teste - baseline_score, scorer.optim_algo, "#results"))

class SGD:
    def __init__(self, lr, budget, dimension, ascent=True,  patience=10, patience_eps=1e-10, lr_annealing=0.99):
        self.patience = patience
        self.patience_eps = patience_eps
        self.lr_annealing = lr_annealing
        self.lr = lr
        self.budget = budget
        self.dimension = dimension
        self.x = np.zeros(dimension)
        self.ascent = ascent
        self.nb_steps = 0

    def optimize(self, scorer):
        best_score = None
        best_x = self.x
        lr = self.lr
        current_patience = 0
        while self.budget > self.nb_steps:
            score = scorer.get_score(self.x)
            if best_score is None or (self.ascent and score > best_score + self.patience_eps) or (not (self.ascent) and score < best_score - self.patience_eps):
                best_x = self.x.copy()
                best_score = score
                current_patience = 0
            else:
                current_patience += 1
                self.x = best_x
                lr/=2
                if current_patience >= self.patience:
                    break

            gradients = scorer.total_gradient
            if not self.ascent:
                gradients = -gradients

            self.x += lr * gradients
            lr *= self.lr_annealing

            self.nb_steps += 1
            print(f"SGD score is {score}. Remaining budget {self.budget}. Ascent = {self.ascent}")
            print(f"Current x is  {self.x}")
        return best_x


class Adam:
    def __init__(self, lr, budget, dimension, ascent=True, beta1=0.9, beta2=0.999, epsilon=1e-10, patience=10, patience_eps=1e-10, lr_annealing=0.99):
        self.patience_eps = patience_eps
        self.patience = patience
        self.lr_annealing = lr_annealing
        self.epsilon = epsilon
        self.beta2 = beta2
        self.beta1 = beta1
        self.lr = lr
        self.budget = budget
        self.dimension = dimension
        self.x = np.zeros(dimension)
        self.ascent = ascent
        self.nb_steps = 0
        self.m = np.zeros(dimension)
        self.v = np.zeros(dimension)

    def optimize(self, scorer):
        best_score = None
        best_x = self.x
        lr = self.lr
        current_patience = 0
        while self.budget > self.nb_steps:
            score = scorer.get_score(self.x)
            if best_score is None or (self.ascent and score > best_score + self.patience_eps) or (not(self.ascent) and score < best_score - self.patience_eps):
                best_x = self.x.copy()
                best_score = score
                current_patience=0
            else:
                current_patience+=1
                lr/=2
                self.x = best_x
                if current_patience >= self.patience:
                    break
            gradients = scorer.total_gradient
            if not self.ascent:
                gradients = -gradients
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
            self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
            self.m /= (1 - self.beta1 ** (self.nb_steps + 1))
            self.v /= (1 - self.beta2 ** (self.nb_steps + 1))
            self.x += self.lr * self.m/(np.sqrt(self.v) + self.epsilon)

            lr *= self.lr_annealing
            self.nb_steps += 1
            print(f"SGD score is {score}. Best score {best_score}. Remaining budget {self.budget}. Ascent = {self.ascent}")
            print(f"Current x is  {self.x}")
            print(f"Current best x is  {best_x}")
        return best_x


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

    partition = "uninterrupted"  # dev

    image_indices = range(num_images)
    penalizations = [1.0]
    budgets = [10000]  # [1000, 10000]
    dims = [dim_per_layer ** 2 * 23 * 3 for dim_per_layer in [20]]
    tools = ["SGD","Adam"]
    discriminator_coeffs = [1.0]
    clamping = [None]
    blur_based_penalizations = [None]

    args_array = itertools.product(image_indices, penalizations, budgets, dims, tools, discriminator_coeffs, clamping,
                                   blur_based_penalizations)
    run_on_cluster = args.run_on_cluster
    if run_on_cluster:
        executor = submitit.AutoExecutor(folder=log_path)
        executor.update_parameters(gpus_per_node=1, cpus_per_task=3, timeout_min=2500, partition=partition,
                                   comment="srgan")
        executor.update_parameters(array_parallelism=80)
        map = executor.map_array(optimize_noise, args_array)
        print(map)
    else:
        for argument in args_array:
            optimize_noise(argument)
