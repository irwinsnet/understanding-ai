---
title: Readme
marimo-version: 0.23.1
width: medium
---

# Understanding AI
Stacy Irwin

Last Updated: 12 April 2026

This repository contains several Marimo notebooks that explain how large language models (LLM) work. The target audience is high school FIRST robotics students who have completed Algebra I and who are familiar with Python, Git, and VS Code.

## Table of Contents
1. Terrminology and Concents: 01-terminology/terminology.md
2. Intro to Machine Learning: 02-machine-learning/machine-learning.md

## Running Marimo Notebooks.
1. Install VS Code. See https://code.visualstudio.com/download.
2. Install Git. See https://git-scm.com/install/windows.
    * Select all default installation options EXCEPT set VS Code as Git's default editor instead of Vim.
3. Install UV per the installation instructions at https://docs.astral.sh/uv/getting-started/installation/.
4. Use Git to clone this repo.
5. From the repo's root folder run `uv sync` to create a virtual environment and install all required dependencies.
6. Activate the virtual environment: `.\venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux or Mac).
7. Run `marimo edit`
8. Select the desired notebook from the marimo management page.

```python {.marimo}
import marimo as mo
```