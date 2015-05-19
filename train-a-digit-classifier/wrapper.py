from __future__ import division
import os
import os.path
import re
import subprocess

__TORCH_DEMOS_HOME__ = os.path.join(
    os.environ['HOME'],
    'torch-demos',
    'train-a-digit-classifier')

os.chdir(__TORCH_DEMOS_HOME__)


# -- parameters
seed = 1
learning_rate = 0.05
momentum = 0.0
ell1 = 0.0
ell2 = 0.0


# -- launch subprocess
out = subprocess.check_output([
    'th train-on-mnist.lua '  +
    '--full '                 +
    '--model=mlp '            +
    '--batchSize=100 '        +
    '--epochs=1 '             +
    '--seed=%d '         % seed          +
    '--learningRate=%f ' % learning_rate +
    '--momentum=%f '     % momentum      +
    '--coefL1=%f '       % ell1          +
    '--coefL2=%f '       % ell2
    ],
    shell=True
)

# -- parse output from stdout
out = out.splitlines()                              # keep last line
out = re.compile(r'\x1b[^m]*m').sub('', out[-1])    # strip color codes
out = out.split('=')[-1]                            # strip left hand side
out = float(out)
