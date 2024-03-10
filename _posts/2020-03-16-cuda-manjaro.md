---
layout: post
title: Installing CUDA 10.1 and cuDNN 7.6 on Manjaro Linux
categories: linux nvidia deep-learning
---

<!-- TOC -->

- [Introduction](#introduction)
- [Uninstall what was wrongfully installed](#uninstall-what-was-wrongfully-installed)
- [Install/downgrade NVIDIA drivers](#installdowngrade-nvidia-drivers)
  - [Restart computer and test drivers](#restart-computer-and-test-drivers)
- [Install CUDA 10.1 and cuDNN 7.6](#install-cuda-101-and-cudnn-76)
  - [CUDA](#cuda)
  - [cuDNN](#cudnn)
  - [Edit `.profile`](#edit-profile)

<!-- /TOC -->

## Introduction

If you use Manjaro Linux and want to use TensorFlow or PyTorch with GPU support, you'll need to install a version of CUDA that works with these libraries. As of the time of this writing, that is CUDA 10.1, which is compatible with cuDNN 7.6.5 and only works under NVIDIA drivers >= 418.39 and < 440.33 [[source]](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver).

Nowadays, the drives installed automatically by Manjaro are only compatible with CUDA 10.2. So, in order for our deep learning packages to work, a downgrade is necessary, both in the video drivers and in CUDA, in case you have installed it.

Let's get to it.

## Uninstall what was wrongfully installed

First, make sure you don't have any version of CUDA or cuDNN installed. If you've installed them via `pacman` (`cuda` and `cudnn` package, respectively), uninstall them before the next steps, or else Manjaro won't let you swap drivers when the time comes.

```shell
sudo pacman -R cuda cudnn
```

## Install/downgrade NVIDIA drivers

If you are already using your video card for something on your system, such as gaming, chances are Manjaro has installed the most recent NVIDIA drivers, which are incompatible with CUDA 10.1 and need to be uninstalled.

At the time of this writing, the preferred package is `video-nvidia-440xx`, which installs version 440.64 of the drivers. Since we're working with Manjaro, I suggest uninstalling the drivers using the the `mhwd` tool, like so:

```shell
sudo mhwd -r pci video-nvidia-440xx
```

This should not make your video card stop working immediately (as in receive a black screen right after this command). At least, it didn't happen to me.

If you don't have video drivers installed or have just uninstalled them, now it's time to install the most recent version of the proprietary drivers compatible with CUDA 10.1. Any version before 440.39 should suffice, but `mhwd` provides 435.21 nicely bundled and that should be enough for us:

```shell
sudo mhwd -i pci video-nvidia-435xx
```

### Restart computer and test drivers

This is a good time to restart the computer and check if it correctly loads the new drivers and detects the NVIDIA graphics card. A good test is to run the `nvidia-smi` command after a reboot and check that the command lists information like this:

```
Mon Mar 16 08:11:11 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1070    Off  | 00000000:01:00.0  On |                  N/A |
| 27%   40C    P0    34W / 151W |     72MiB /  8117MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+


+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      9630      G   /usr/lib/Xorg                                 72MiB |
+-----------------------------------------------------------------------------+
```

## Install CUDA 10.1 and cuDNN 7.6

### CUDA

`pacman` provides CUDA 10.2 and upwards in package `cuda`, but we need version 10.1, which we'll install from the AUR:

```shell
yay -S cuda-10.1
```

### cuDNN

cuDNN is available via pacman as `cudnn`, but, even though it lists `cuda>=10` as a dependency and we've just installed `cuda-10.1`, it doesn't recognize the requirement as met (probably because it came from the AUR) and tries to install the pacman `cuda` package, whose version is 10.2. In order to bypass that:

1. go to the AUR page for the cudnn package. The commit we want (containing version 7.6) is [this one](https://git.archlinux.org/svntogit/community.git/tree/repos/community-x86_64?h=packages/cudnn&id=c8950411997c56322bb06bb1388189401911287a);
2. download the PKGBUILD and accompanying PDF file;
3. edit the PKGBUILD file, removing the line saying `depends=('cuda>=10')`;
4. inside the directory with the edited PKGBUILD file and accompanying PDF file, use the commands:

```shell
makepkg
makepkg --install
```

Now you should have cuDNN installed.

### Edit `.profile`

Add the following lines to your `.profile` file, appending the location of the CUDA executables and include libraries and provided libraries to `PATH`, `CPATH` and `LD_LIBRARY_PATH`, respectively:

```shell
export PATH="/opt/cuda-10.1/bin:$PATH"
export CPATH="/opt/cuda-10.1/include:$CPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cuda-10.1/lib64:/opt/cuda-10.1/lib"
```

Logout and login back again for the changes in `.profile` to take effect.

---

Now you should be all setup to use CUDA 10.1 and cuDNN natively on your Manjaro computer. I have successfully used this procedure both for TensorFlow and PyTorch. Happy coding!
