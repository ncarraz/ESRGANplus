#!/bin/bash
cd /private/home/broz/workspaces/tests_malagan/malagan
pushd codes
touch toto
rm toto
echo $* > toto
sed -i "s/sigma=[0-9\.]*/sigma=0.0001/g" models/modules/RRDBNet_arch.py
sed -i "s/zeseed=['ot0-9\.]*/zeseed='toto'/g" models/modules/RRDBNet_arch.py
touch zeres
rm zeres
python3.6 get_koncept_512_score.py -opt options/test/test_ESRGAN.yml 2>&1 | cat > zeres
echo $SEED ' ' $SIGMA ' ' `cat zeres | grep -i Concept512Score |tail -n 1`

popd
echo `cat codes/zeres | grep -i Concept512Score | tail -n 1` | sed ' s/.*Concept512Score://g'
