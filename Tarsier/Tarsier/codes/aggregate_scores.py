import os
import numpy as np

base_folder = "/Users/broz/workspaces/tests_malagan/malagan/results/ESRGANplus_x4/"
folder_set14 = os.path.join(base_folder, "set14/set14_log_koncept_CMA_pessimistic_scores_penal2_disc_001/")
folder_set5 = os.path.join(base_folder, "set5/set5_log_koncept_OnePlusOne_CMA_pessimistic_scores_grid_search/")
folder_flickr = os.path.join(base_folder, "flickr/flickr_log_koncept_OnePlusOne_CMA_pessimistic_scores_grid_search/")
folder_PIRM_test = os.path.join(base_folder, "PIRM_Test/PIRM_test_CMA/")
folder_PIRM_validation = os.path.join(base_folder, "PIRM_Val/PIRM_test_CMA/")

suffix_baseline = "_koncept_512_Baseline.png.txt"
suffix_penal1 = "_koncept_512_DiagonalCMA_penal1.0_27600_disc0.1_clamping2.5_bud10000.png.txt"
suffix_penal2_disc001 = "_koncept_512_DiagonalCMA_penal2.0_27600_disc0.01_clamping2.5_bud10000.png.txt"

# prefix = "/Users/broz/workspaces/tests_malagan/malagan/results/ESRGANplus_x4/set14/set14_log_koncept_CMA_pessimistic_scores_penal2_disc_001/Perceptual_Score"
results = {}
folders = [folder_set5, folder_set14, folder_flickr, folder_PIRM_test, folder_PIRM_validation]

for folder in folders:
    prefix = folder + "Perceptual_Score"
    results[folder] = {}
    suffixes = [suffix_baseline, suffix_penal1, suffix_penal2_disc001]
    for suffix in suffixes:
        results[folder][suffix] = []
        try:
            with open(prefix + suffix) as f:
                for line in f:
                    perceptual_score = float(line.split(":")[-1])
                    Ma_score = float(line.split(",")[0].split(':')[-1])
                    NIQE_score = float(line.split(",")[1].split(':')[-1])
                    # print(perceptual_score)
                    results[folder][suffix].append(perceptual_score)
        except FileNotFoundError:
            continue
    print()
    for suffix in suffixes:
        print(prefix.split('/')[-3], suffix, ':', np.array(results[folder][suffix]).mean(), ", images:", len(results[folder][suffix]))


results_PIRM_test = results[folder_PIRM_test]
relative_results = np.array(results_PIRM_test[suffix_penal1]) - np.array(results_PIRM_test[suffix_baseline])

for i, res in enumerate(relative_results):
    print(201 + i, res)