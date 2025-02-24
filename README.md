<!--
 * @Author: Chris Xiao yl.xiao@mail.utoronto.ca
 * @Date: 2024-03-31 01:27:47
 * @LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
 * @LastEditTime: 2024-03-31 02:12:20
 * @FilePath: /CPD-Pytorch/README.md
 * @Description: Readme file
 * I Love IU
 * Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
-->
![CPD-Pytorch](https://socialify.git.ci/mikami520/CPD-Pytorch/image?description=1&font=Source+Code+Pro&forks=1&issues=1&language=1&name=1&owner=1&pattern=Floating+Cogs&pulls=1&stargazers=1&theme=Light)
# CPD-Pytorch
Coherent Point Drift Implementation in pytorch version

# Installation
```bash
git clone https://github.com/mikami520/CPD-Pytorch.git
cd CPD-Pytorch
pip install -e .
```

# Example Usage
```python
from functools import partial
import matplotlib.pyplot as plt
from torchcpd import RigidRegistration
import numpy as np
import torch as th

device = 'cuda:0' if th.cuda.is_available() else 'cpu'
X = np.loadtxt('data/bunny_target.txt')
# synthetic data, equaivalent to X + 1
Y = np.loadtxt('data/bunny_source.txt')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

callback = partial(visualize, ax=ax, fig=fig, save_fig=False)

reg = RigidRegistration(**{'X': X, 'Y': Y, 'device': device})
reg.register(callback)
plt.show()
```
**More tutorials can be found in the ```/examples``` folder.**

# Star History

<p align="center">
  <a href="https://star-history.com/#mikami520/CPD-Pytorch&Date">
   <picture>
     <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mikami520/CPD-Pytorch&type=Date&theme=dark" />
     <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mikami520/CPD-Pytorch&type=Date" />
     <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mikami520/CPD-Pytorch&type=Date" />
   </picture>
  </a>
</p>
