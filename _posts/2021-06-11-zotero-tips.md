---
layout: post
title: Zotero tips and tricks
categories: zotero tutorial
---

**tl;dr:** [**Zotero**][zotero] is great but it lacks good cloud sync support and a way to make a giant .bib file available for referencing. You can get the first with the [**Zotfile**][zotfile] addon, pointing its PDF directory somewhere inside your cloud sync directory. You can get the second with the [**Better BibTeX**][better-bibtex] addon, by exporting a `.bib` file which is always kept updated.

## Background

[**Zotero**][zotero] is a great reference manager that has incorporated lots of much needed features along the years. Two things that I feel Zotero is still missing is

- good cloud sync support, so you can keep as many PDF files in your libraries without going over the measly 300 MB quota it gives;
- a central `.bib` file which can be referenced in all my projects.

Here, I teach you how to circumvent these problems with two addons.

## Cloud sync with Zotfile

[**Zotfile**][zotfile] is an addon that helps in managing PDF files inside Zotero. It allows for automatic and batch moving and renaming of file attachments.

In order to sync all your PDF files among multiple computers, you can point Zotfile's "PDF directory" setting into a folder inside your cloud sync service of choice (Mega, Dropbox, Google Drive, OneDrive etc.) and configure the addon to automatically move all new attachments to this folder. Zotero will then only keep links to these files in its database and will synchronize only these links, while the files themselves will be synchronized by your cloud sync application.

## Central up-to-date bib database with Better BibTeX

[**JabRef**][jabref] introduced me to the as-of-yet unknown habit of keeping a single giant `.bib` file, which I would just link all my papers to and use the autocomplete function of my TeX editor of choice to search for entry keys.

To emulate this behavior in Zotero, there is an addon called [**Better Bibtex**][better-bibtex]. It allows me to export my entire library into a `.bib` file, which is then kept updated as I change information in my Zotero library. I actually kept two `.bib` files, one in BibTeX format for all my papers, and the other in BibLaTeX format, for my thesis.

[zotfile]: http://zotfile.com/
[jabref]: https://www.jabref.org/
[zotero]: https://www.zotero.org/
[better-bibtex]: https://retorque.re/zotero-better-bibtex/
