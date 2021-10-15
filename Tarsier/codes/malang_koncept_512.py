import sys
sys.path = sys.path = ["/private/home/broz/workspaces/nevergrad"] + sys.path
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--opt_path", help="Path to the .yml options file", default="options/test/test_ESRGAN.yml")
parser.add_argument("--penalization", help="penalization factor", default=5e-3, type=float)
parser.add_argument("--discriminator_coeff", help="penalization factor", default=0, type=float)
parser.add_argument("--gpu_id", help="GPU id", default=0, type=int)
parser.add_argument("--dim_per_layer", help="SQRT of noise dim per layer", default=5, type=int)
parser.add_argument("--budget", help="", default=1000, type=int)
parser.add_argument("--optimizer", help="", type=str, default="DiagonalCMA")
parser.add_argument("--penalisation_mode", help="", type=str, default="l2")
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
from scipy.stats import norm

loggers = {}

class NNScorer:
    def __init__(self, opt_path_, penalisation=1.0, image_index=0, penalisation_mode='l2', discriminator_coeff=0):
        self.discriminator_coeff = discriminator_coeff
        global loggers
        self.image_index = image_index
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
        self.dataset_dir = osp.join(self.opt['path']['results_root'], test_set_name)
        util.mkdir(self.dataset_dir)
        self.quality_scorer = Koncept512Predictor(self.dataset_dir,
                                                  aux_root="/private/home/broz/workspaces/tests_malagan/malagan/codes/koniq/")

        self.model = create_model(self.opt)

    def get_score(self, toto):

        test_start_time = time.time()

        for i, data in enumerate(self.test_loader):
            if i != self.image_index:
                continue

            self.need_GT = False if self.test_loader.dataset.opt['dataroot_GT'] is None else True
            self.model.feed_data(data, need_GT=self.need_GT)
            img_path = data['GT_path'][0] if self.need_GT else data['LQ_path'][0]
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
            image_name = self.img_name + "_koncept_512_" + self.optim_algo + (suffix if suffix else '') + '.png'
            save_img_path = osp.join(self.dataset_dir, image_name)
            self.logger.info("Saving image in " + save_img_path)
            util.save_img(self.sr_img, save_img_path)

            # test_results = OrderedDict()
            # test_results['psnr'] = []
            # test_results['ssim'] = []
            # test_results['psnr_y'] = []
            # test_results['ssim_y'] = []
            # self.compute_psnr_ssim(test_results, prefix=optim_algo)
            score = self.quality_scorer.predict(image_name, repeats=1)
            l2 = self.penalisation_lambda * (np.array(toto) ** 2).mean()

            fake_image_dim = self.model.fake_H.size()
            print("fake image dimension:{}".format(fake_image_dim))
            dim_x, dim_y = fake_image_dim[2], fake_image_dim[3]
            nb_x = dim_x // 128
            nb_y = dim_y // 128
            discriminator_preds = []
            if self.discriminator_coeff:
                for i in range(nb_x):
                    for j in range(nb_y):
                        fake_image_patch = self.model.fake_H[:, :, i * 128:(i+1) * 128, j * 128: (j+1) * 128]
                        with torch.no_grad():
                            discriminator_preds.append(self.model.netD(fake_image_patch).cpu().numpy()[0][0])
            else:
                discriminator_preds.append(0)
            discriminator_preds = np.array(discriminator_preds)
            print('Discriminator scores: {}'.format(discriminator_preds))
            discriminator_score = self.discriminator_coeff * discriminator_preds.mean()
            self.logger.info(
                '{} Koncept512Score:{:.6f}, Penalization: {}, Discriminator score: {},  image name {}'.format(i, score, l2, discriminator_score, image_name))
            return score - l2 + discriminator_score

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


def malagan(x):
    print("#toto parametrisation is {}".format(x))
    return -scorer.get_score(x)

dim_per_layer = args.dim_per_layer ** 2
dim = dim_per_layer * 23 * 3
if __name__ == '__main__':
    # image_index = 1
    opt_path = args.opt_path
    for image_ in range(20):
        baseline_score = 0
        computed_baseline = False
        for penalization_lambda in [0.6]: #[ 5e-4, 8e-4, 1e-3, 2.5e-3, 5e-3]:#[args.penalization]:  # , 1e-2]:#[1e-5, 1e-3, 1e-1, 1]:
            for budget in [1000]:
                for dim_per_layer in [5, 30]:
                    dim = dim_per_layer ** 2 * 23 * 3
                    scorer = NNScorer(opt_path, penalisation=penalization_lambda, image_index=image_, penalisation_mode=args.penalisation_mode)
                    scorer.discriminator_coeff = args.discriminator_coeff
                    for tool in ["OnePlusOne", "DiagonalCMA", "DDE"]:#, "OnePlusOne", "DiagonalCMA"]:#[args.optimizer]:  # , "RandomSearch", "TwoPointsDE", "DE", "PSO", "SQP"]:
                        if not computed_baseline:
                            print("The baseline hasn't been computed yet. Computing it")
                            scorer.optim_algo = "Baseline_{}_{}_{}".format(budget, penalization_lambda, dim)
                            baseline_optimizer = ng.optimizers.registry["OnePlusOne"](instrumentation=dim, budget=1)
                            recommendation = baseline_optimizer.optimize(malagan)
                            teste = malagan(recommendation.data)
                            baseline_score = teste
                            print("The baseline score is {}".format(baseline_score))
                            computed_baseline = True
                        scorer.optim_algo = "{}_bud{}_penal{}{}_{}_disc{}".format(tool, budget, 'g' if args.penalisation_mode == "gaussian" else '', penalization_lambda, dim, scorer.discriminator_coeff)
                        optimizer = ng.optimizers.registry[tool](instrumentation=dim, budget=budget)
                        recommendation = optimizer.optimize(malagan)
                        print("#recommendation {}".format(recommendation))  # optimal args and kwargs
                        teste = malagan(recommendation.data)
                        print(budget, tool, "Score:", teste, teste - baseline_score, "#results")

    # ./fitness_test.sh 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # ~/malagan/codes ~/malagan
    #    odd PSNR_Y: 26.282498 dB; SSIM_Y: 0.661995   even PSNR_Y: 23.751754 dB; SSIM_Y: 0.677056
    #    PSNR to be maximized
    #    SSIM to be maximized
    #    ~/malagan
    #     23.751754
