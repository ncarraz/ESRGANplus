#!/bin/bash
cd /private/home/broz/workspaces/malagan
pushd codes
touch toto
rm toto
echo $* > toto
sed -i "s/sigma=[0-9\.]*/sigma=0.0001/g" models/modules/RRDBNet_arch.py
sed -i "s/zeseed=['ot0-9\.]*/zeseed='toto'/g" models/modules/RRDBNet_arch.py
touch zeres
rm zeres
python3.6 test.py -opt options/test/test_ESRGAN.yml 2>&1 | cat > zeres
echo $SEED ' ' $SIGMA ' ' `cat zeres | grep -i odd | grep -i PSNR_Y | tail -n 1` ' ' `cat zeres | grep -i even | grep -i PSNR_Y | tail -n 1`

echo PSNR to be maximized
echo SSIM to be maximized
popd
echo `cat codes/zeres | grep -i odd | grep -i PSNR_Y | tail -n 1` | sed ' s/.*PSNR_Y://g' | sed 's/dB.*//g'
