---
layout: post
title: Mapping Numpad keys with sxhkd
categories: linux keyboards
---

In this tutorial, I explore how to map the Numpad keys to `sxhkd` in a Cooler Master Masterkeys Pro M ABNT2 keyboard.

<!-- TOC -->

- [About `sxhkd`](#about-sxhkd)
- [About the keyboard](#about-the-keyboard)
- [Getting keysims with `xev` when Num Lock is OFF...](#getting-keysims-with-xev-when-num-lock-is-off)
- [... and when Num Lock is turned ON](#-and-when-num-lock-is-turned-on)
- [Testing what works in `sxhkd`](#testing-what-works-in-sxhkd)
    - [Num Lock OFF](#num-lock-off)
    - [Num Lock ON](#num-lock-on)
- [Final Notes](#final-notes)

<!-- /TOC -->

## About `sxhkd`

If you're here, I supposed you know what [`sxhkd`](https://github.com/baskerville/sxhkd) is, but in case you don't, it is a Linux program that allows commands to be triggered by X input events. Basically, we use it to create OS-wide keyboard/mouse shortcuts.

`sxhkd` is especially useful in a window manager such as [`bspwm`](https://github.com/baskerville/bspwm/), which can be fully controlled via the command line.

The shortcuts are configured through a configuration file called `sxhkdrc`, which is kept in the `$XDG_CONFIG_HOME/sxhkd` directory. An example of this file can be found [here](https://github.com/baskerville/bspwm/blob/master/examples/sxhkdrc).

## About the keyboard

According to the [CoolerMaster product website](https://www.coolermaster.com/catalog/peripheral/keyboards/masterkeys-pro-m-white), the Masterkeys Pro M is a 90% keyboard, which means it has a Numpad, with the navigation keys merged with it.

**When Num Lock is on**, we have access to **numbers, decimals, separators and operators** when the keys are pressed.

**When Num Lock is off**, we have access to **Insert, Delete, Home, End, Page Up, Page Down, Print Screen, Scroll Lock, Pause and arrows**, using the same keys.

To complicate things a bit more, my keyboard uses the ABNT2 standard and that makes the Numpad a little different too. The figure below shows what I am working with (taken from [this article](https://www.phe.com.br/03/06/2017/cooler-master-masterkeys-pro-m/)). A link to the standard English Numpad can be found [here](https://imgur.com/a/BlO5Iw3) for comparison.

![Masterkeys Pro M ABNT2 Numpad](https://www.phe.com.br/wp-content/uploads/2017/06/DSC_0601-1-1024x683.jpg)

## Getting keysims with `xev` when Num Lock is OFF...

These are the keysims that the `xev` Linux program gave me when pressing each one with Num Lock turned off.

To get all keysims correctly, I had to unbind the shortcuts that were already present in some keys, such as **Print Screen**, as they were triggering other X events.

Also, using the `-event keyboard` option helped me filter only the relevant events.

    Num_Lock  Print Scroll_Lock   Pause
     Insert   Home     Prior      KP_Add
     Delete    End      Next    KP_Decimal
    [Nothing]  Up    [Nothing]   KP_Enter
      Left    Down     Right     KP_Enter

Keys 1 and 3 don't send any events at all, so I believe the keyboard just sends no signals for those keys when Num Lock is off.

## ... and when Num Lock is turned ON

When Num Lock is on, there are no surprises. Every key has a key name on `xev`.

    Num_Lock KP_Divide KP_Multiply  KP_Subtract
      KP_7     KP_8       KP_9        KP_Add
      KP_4     KP_5       KP_6      KP_Decimal
      KP_1     KP_2       KP_3       KP_Enter
      KP_0     KP_0    KP_Separator  KP_Enter

## Testing what works in `sxhkd`

In order to test the usage of these keysims in `sxhkd`, I created the following shortcuts. Basically, when I press `super` + a key in the Numpad, I should get its name in a system notification.

    super + KP_{0-9,Divide,Multiply,Subtract,Add,Decimal,Enter,Separator}
        notify-send {0-9,Divide,Multiply,Subtract,Add,Decimal,Enter,Separator}

    super + {Num_Lock,@Print,Insert,Home,Prior,Delete,End,Next,Scroll_Lock,Pause}
        notify-send {Num_Lock,Print,Insert,Home,Prior,Delete,End,Next,Scroll_Lock,Pause}

Below I show which keysims worked.

### Num Lock OFF

    Num_Lock KP_Divide KP_Multiply  KP_Subtract
      KP_7     KP_8        KP_9       KP_Add
      KP_4     KP_5        KP_6     KP_Decimal
      KP_1     KP_2        KP_3      KP_Enter
      KP_0     KP_0    KP_Separator  KP_Enter

1. Even though I didn't map the arrow keys in the above shortcut, I already knew they worked from countless other shortcuts that use them, so I added them to final results.

2. At first, When I pressed **Scroll Lock** I was getting a notification that **Pause** was being pressed, but when I read the sxhkd man page, its said that

   >If you have a non-QWERTY keyboard or a non-standard layout configuration, you should provide a `COUNT` of `1` to the `-m` option or `-1` (interpreted as infinity) if you constantly switch from one layout to the other [...]"

   So, when I passed `1` to `-m`, the **Scroll Lock** and **Pause** keys started working as expected.

### Num Lock ON

    Num_Lock KP_Divide KP_Multiply  KP_Subtract
                                      KP_Add
                                    KP_Decimal
                                     KP_Enter
                                     KP_Enter

All of the **number keys** + **the separator key** (a comma, in my case) did not work. The keysims of the three keys in the top row changed, so they could potentially be reused to other sxhkd shortcuts.

## Final Notes

Even when `Num_Lock` and `Scroll_Lock` were mapped in `sxhkdrc`, pressing them alongside a shortcut still switched their respective lock modes on/off, which means that using these keys in keyboard shortcuts may be a bad idea, as one would be switching them on and off uncontrollably, changing the behavior of other keys and programs in the process.

So my recommendation is to leave them alone (as well as Caps Lock).

Another thing. If I:

1. pressed `super + Num_Lock`
2. released `Num_Lock`, but kept `super` pressed
3. pressed one of the following keys, which had its keysim swapped once the Num Lock mode switched on/off:

   - `@Print`/`KP_Divide`
   - `Scroll_Lock`/`KP_Multiply`
   - `Pause`/`KP_Subtract`

I would still get *the keysim from the active Num Lock mode at the moment `super` was first pressed*. In order to "refresh" the Num Lock mode for `sxhkd`, I needed to release `super` and press it again.
