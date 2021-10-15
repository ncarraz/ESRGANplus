#!/bin/bash
export log_path=logs/set5_SGD_ADAM #logs/PIRM_VAL_SGD_ADAM #logs/PIRM_CMA_Val_penal_1_blur_penalizations #export log_path=logs/PIRM_TEST_SGD_ADAM/
export dataset=PIRM_Val
for method in SGD Adam; do #Adam SGD
  for blur in None; do
    for body in _penal1.0_27600_disc1.0_clampingNone_bud10000_blur_; do
#      for image in baby bird butterfly head woman; do #{1..100}; do #for image in {201..300}; do
      for image in {1..100}; do #for image in {201..300}; do
        cat $log_path/*/*.err | grep ${image}_ | grep Koncept512Score | grep _koncept_512_${method}${body}${blur}.png| tail -1;  done > scores_${dataset}_koncept_512_${method}${body}${blur}.txt; echo scores_${dataset}_koncept_512_${method}${body}${blur}.txt; cat scores_${dataset}_koncept_512_${method}${body}${blur}.txt | awk '{print $13}' | sed 's/,//' | awk '{ total += $1; count++ } END { print total, count, total/count }'; done;done;done
