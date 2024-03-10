---
layout: post
title: Using task-spooler to queue experiments on Linux
categories: linux deep-learning tutorial
---

In this post, I'll introduce you to [task-spooler](http://viric.name/soft/ts/) (“ts” for short), a Linux program that lets you queue tasks to be executed either sequentially or in parallel, according to a user-defined number of slots.

<!-- TOC -->

- [Problem setting](#problem-setting)
- [A crude solution](#a-crude-solution)
- [The downsides](#the-downsides)
- [Enter ts](#enter-ts)
- [Installing ts](#installing-ts)
- [Closing remarks](#closing-remarks)

<!-- /TOC -->

## Problem setting

Let's say you have a set of experiments to execute on your Linux machine. Each experiment is called via the python command and a script is passed, with some accompanying command-line args. For example, experiment 1 can be called like so:

``` sh
python train.py --network 1 --dataset 1
```

Now let's say there are 5 networks you want to test in 4 different datasets, for a total of 20 experiments. You also know that you can run 3 of these experiments at the same time in a single machine with little to no loss of performance. The question is: what's an efficient way to automatically execute all of these experiments?

## A crude solution

A crude solution may involve creating 3 shell scripts, or 3 shell functions, which encapsulate equal-sized batches of experiments. For example, one way to go about organizing our experiments is the following.

Script 1:

``` sh
python train.py --network 1 --dataset 1
python train.py --network 1 --dataset 2
[...]
python train.py --network 2 --dataset 3
```

Script 2:

``` sh
python train.py --network 2 --dataset 4
python train.py --network 3 --dataset 1
[...]
python train.py --network 4 --dataset 2
```

Script 3:

``` sh
python train.py --network 4 --dataset 3
python train.py --network 1 --dataset 2
[...]
python train.py --network 5 --dataset 4
```

You then run each script/function in a dedicated terminal, maybe using tmux/byobu, and log out of the machine, leaving the experiments to finish.

## The downsides

One downside of this method is that the computation load has been manually balanced from the start and, if one of our manual queues finish before the others, that queue will not be able to get tasks that are waiting to be executed in our other scripts. This may result in idle computation resources.

Another downside of this method is that there is no way to prioritize which experiments should be executed first. We can take an educated guess and say that the first experiment of each manual queue will finish first, then the second experiment of each manual queue and so on. But this method is unpredictable and, with time, the estimate of which experiments are still running or when a specific experiment will end may diverge.

The solution to these issues would be to have an easy way to queue all experiments, tell the computer how many of them can be executed in parallel and let it consume the queue of experiments automatically.

## Enter ts

`ts` works with the concept of _slots_. You tell it the maximum number of slots it has to run processes and, when queuing a process, how many slots each process requires. By default, `ts` provides a single available slot and each new process also requires one slot by default, so a single task will be executed every time.

In order to add a command to the `ts` queue, we just have to call the command preceeded by the `ts` command. For example, in order to add our experiments to the queue, we would call them like so:

``` sh
ts python train.py --network 1 --dataset 1
ts python train.py --network 1 --dataset 2
ts python train.py --network 1 --dataset 3
[...]
ts python train.py --network 5 --dataset 4
```

Or, if you want to get fancy:

``` sh
for i in $(seq 1 5); do
  for j in $(seq 1 4); do
    ts python train.py --network $i --dataset $j
  done
done
```

Since `ts` provides a single slot by default, only one of the experiments will start executing. The others will stay in the queue and the current terminal will not hang.

Please note that, in some systems, the actual program may be called `tsp` due to a naming conflict with some other famous Linux program.

### Adding more slots to the ts queue

In our problem, it seems suitable to tell `ts` that it can execute 3 tasks simultaneously, so we go ahead and do that through the `-S` flag.

``` sh
ts -S 3
```

This way, `ts` will fill the two newly created slots by running two additional tasks.

### Adding more demaning tasks to the queue

If we have a more demanding task that we want to add to the queue, but only be executed when more than one slot is available in the queue, we can use the `-N` to configure how many available slots `ts` needs to start the task.

```sh
ts -N 3 python train_big_net.py
```

The task above will only start once our whole 3-slot queue is empty. If there are other tasks in the queue that would take less than 3 slots to start, they will not be started, since (by definition) tasks in the queue are executed sequentially.

### Monitoring tasks

In order to check on the list of processes and their statuses, just call `ts` with no arguments. In the example output below, I have called `ts sleep 60` a bunch of times with a number of 4 slots:

``` 
ID   State      Output               E-Level  Times(r/u/s)   Command [run=4/4]
12   running    /tmp/ts-out.AEnsRv                           sleep 60
13   running    /tmp/ts-out.uDzxH8                           sleep 60
14   running    /tmp/ts-out.pGUSbK                           sleep 60
15   running    /tmp/ts-out.MWfo2l                           sleep 60
16   queued     (file)                                       sleep 60
17   queued     (file)                                       sleep 60
18   queued     (file)                                       sleep 60
19   queued     (file)                                       sleep 60
20   queued     (file)                                       sleep 60
0    finished   /tmp/ts-out.UPIohv   0        60.00/0.00/0.00 sleep 60
1    finished   /tmp/ts-out.l2bBJq   0        60.00/0.00/0.00 sleep 60
6    finished   /tmp/ts-out.4SZ7O2   0        60.00/0.00/0.00 sleep 60
7    finished   /tmp/ts-out.3qkpyK   0        60.00/0.00/0.00 sleep 60
8    finished   /tmp/ts-out.ngun7M   0        60.00/0.00/0.00 sleep 60
9    finished   /tmp/ts-out.Fb5QLU   0        60.00/0.00/0.00 sleep 60
10   finished   /tmp/ts-out.lLGHom   0        60.00/0.00/0.00 sleep 60
11   finished   /tmp/ts-out.0B8fyB   0        60.00/0.00/0.00 sleep 60
```

You can see there are 4 processes running, 5 queued and 8 have finished after having been running for (unsurprisingly) 60 seconds.

In order to view the output of one of the tasks, there are commands such as `ts -t <task_id>` and `ts -c <task_id>`.

### Miscellaneous functionalities

For completion sake, here is the relevant output of `ts -h`:

```
Actions:
  -K       kill the task spooler server
  -C       clear the list of finished jobs
  -l       show the job list (default action)
  -S [num] get/set the number of max simultaneous jobs of the server.
  -t [id]  "tail -n 10 -f" the output of the job. Last run if not specified.
  -c [id]  like -t, but shows all the lines. Last run if not specified.
  -p [id]  show the pid of the job. Last run if not specified.
  -o [id]  show the output file. Of last job run, if not specified.
  -i [id]  show job information. Of last job run, if not specified.
  -s [id]  show the job state. Of the last added, if not specified.
  -r [id]  remove a job. The last added, if not specified.
  -w [id]  wait for a job. The last added, if not specified.
  -k [id]  send SIGTERM to the job process group. The last run, if not specified.
  -u [id]  put that job first. The last added, if not specified.
  -U <id-id>  swap two jobs in the queue.
  -B       in case of full queue on the server, quit (2) instead of waiting.
  -h       show this help
  -V       show the program version
Options adding jobs:
  -n       don't store the output of the command.
  -E       Keep stderr apart, in a name like the output file, but adding '.e'.
  -g       gzip the stored output (if not -n).
  -f       don't fork into background.
  -m       send the output by e-mail (uses sendmail).
  -d       the job will be run only if the job before ends well
  -D <id>  the job will be run only if the job of given id ends well.
  -L <lab> name this task with a label, to be distinguished on listing.
  -N <num> number of slots required by the job (1 default).
```

## Installing ts

Now that I have got your attention, it is time I teach you how to install `ts` . In some Linux distributions, ts is available either via official or unofficial repositories. For example, in Arch-based systems, you can find it in the AUR:

``` sh
yay -S task-spooler
```

In Debian:

``` sh
sudo apt-get install task-spooler
```

However, it is not available in all distros, but it is quite easy to download source from [the developer's website](http://viric.name/soft/ts/) and do the good old untar/ `make` / `make install` combo.

## Closing remarks

`ts` is an extremely simple Linux program that follows the Unix philosophy to the letter: it does a single job and does it well. There are many other functionalities I haven't covered here, but you can cancel queued tasks, consult the output of running and queued tasks and so on.

I understand that there may be a fancier solution to this problem, probably using enterprise-grade software that I an unfamiliar with. However, since most small-time researchers, grad students and hobbyists will really only be using the Linux command-line to ssh on a remote server or run local experiments, I believe having `ts` in the toolbelt is very beneficial.
