---
layout: post
title: Configuring Visual Studio Code for LaTeX
categories: vscode tutorial
---

[VS Code](https://code.visualstudio.com/) has very poor LaTeX support out-of-the-box. Here I list useful extensions and other software that will transform VS Code into a much more power LaTeX editor. They also do not conflict with each other, which is great.

## Getting it out of the way: LaTeX Workshop

While I have used LaTeX Workshop for a long time, I started experiencing a few issues with it that, over time, got really annoying. The main issue is simply [the Enter key not working](https://github.com/James-Yu/LaTeX-Workshop/issues/1193), which was then replaced by [a substantial delay between pressing the Enter key and actually inserting a new line character](https://github.com/James-Yu/LaTeX-Workshop/issues/903). Fixing one apparently introduced the other, so, after many tries, I just gave up on LaTeX Workshop.

## What I use now

- Code completion: [TeXLab VS Code extension](https://marketplace.visualstudio.com/items?itemName=efoerster.texlab). It depends on another program called [TeXLab](https://texlab.netlify.app/), but the extension usually installs it automatically.
- Syntax highlighting [Rich LaTeX syntax highlighting (for use with Texlab)](https://marketplace.visualstudio.com/items?itemName=vomout.latex-syntax).
- Grammar checking: [LTeX](https://marketplace.visualstudio.com/items?itemName=valentjn.vscode-ltex). Be sure to check the [documentation](https://valentjn.github.io/vscode-ltex/docs/settings.html) to learn how to configure the extension to check different languages.
- PDF visualization: If TeXLab has not worked out for you when opening PDF files, you can use an extension called [Open](https://marketplace.visualstudio.com/items?itemName=sandcastle.vscode-open), which opens a file in its default application.

The TeXLab VS Code extension also provides other useful features, as long as the appropriate programs are installed and in your Linux `PATH` (or whatever the equivalent in Mac/Windows is). These assorted programs are usually included in TeX Live, but can be installed through your distribution's package manager, if you are on Linux.

- LaTeX formatting (with `latexindent`)
- Compilation via the F5 shortcut (with `latexmk`)
- Linting (with `chktex`)
- Opening the final PDF file with a shortcut (through the [forward search](https://github.com/latex-lsp/texlab/blob/master/docs/previewing.md) option)
