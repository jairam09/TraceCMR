# Setting up on your laptop (first time)

Do these once, in order. Each step ends with a check so you know it worked. If a step fails, copy the exact error message and ask.

## 1. Install the tools

1. **Python via Miniconda.** Download Miniconda for your operating system from the Anaconda website and install it with the defaults.
   - Check: open a new terminal (Mac: Terminal; Windows: "Anaconda Prompt") and run `conda --version`. You should see a version number.
2. **Git.** Mac: run `git --version`; if it is missing, macOS offers to install it. Windows: install "Git for Windows."
   - Check: `git --version` prints a version.
3. **VS Code**, then inside it install the **Python** and **Jupyter** extensions (Extensions icon on the left sidebar).
4. **GitHub account**, if you do not have one. Also apply for the GitHub Student Developer Pack with your Cornell email.

## 2. Create the project environment

An environment is a separate box of Python packages just for this project, so nothing clashes with anything else on your computer.

```bash
conda create -n mcmr python=3.11 -y
conda activate mcmr
```

- Check: your terminal prompt now starts with `(mcmr)`. **Run `conda activate mcmr` every time you open a new terminal to work on this.**

## 3. Get the code onto your computer

Unzip `mouse-cmr-pipeline.zip` somewhere sensible, for example a `Projects` folder. Then in the terminal:

```bash
cd path/to/Projects/mouse-cmr-pipeline
pip install -e ".[dev]"
```

`-e` means "editable": when you change the code, the changes take effect immediately without reinstalling.

- Check: `pytest` prints `56 passed, 3 skipped` (the skipped tests run only on real UCL data; see docs/datasets.md).
- Check: `python scripts/check_dataset.py --phantom --out data/checks` prints volumes and saves `data/checks/phantom_overlay.png`. Open the PNG: green and red circles should sit exactly on the edges of the gray ring and white disc.

## 4. Put it on GitHub

1. Fill in your full name in `LICENSE` (replace `<YOUR FULL NAME>`), and save.
2. On github.com, click **New repository**, name it `mouse-cmr-pipeline`, choose **Private** for now, and do **not** add a README, license or .gitignore (the project already has them). Click **Create**.
3. GitHub shows commands under "push an existing repository." In your terminal, inside the project folder, run:

```bash
git add LICENSE
git commit -m "Add name to license"
git remote add origin https://github.com/<your-username>/mouse-cmr-pipeline.git
git branch -M main
git push -u origin main
```

(The project already has a first commit. If `git` asks who you are, run the two `git config --global user.name/user.email` commands it suggests, then commit again.)

- Check: refresh the GitHub page and you see the files. Click the **Actions** tab: the tests run automatically and should turn green within a couple of minutes.

## 5. Make it public (after Phase 1 works)

When you can load a real public dataset and the overlays look right: on GitHub go to **Settings → General → Danger Zone → Change visibility → Public**. This starts JOSS's six-month public-history clock.

## Daily routine

```bash
conda activate mcmr
cd path/to/mouse-cmr-pipeline
git pull                 # get any changes (e.g. made on another computer)
# ...work...
pytest                   # before committing, make sure nothing broke
git add -A
git commit -m "Short description of what changed"
git push
```

Commit small working steps often. Write a line in `docs/lab_notebook.md` each session.
