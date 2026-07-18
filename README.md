# Image Editing Without Degradation
### Weon Technical Task 4

When you ask an AI model to edit a photo, it does not patch the one thing
you asked for. It redraws the entire image, and the parts nobody asked to
change come back a little worse every time. This repo reproduces that
failure on a real fashion photoshoot photo, tests seven different fixes
against it, and measures the damage with reference-based image metrics
instead of eyeballing it.

**Headline finding:** diff compositing (let the model edit normally, then
paste original pixels back everywhere nothing actually changed) cut
collateral damage from 69% down to under 1%, more than a hundred times
less than doing nothing. Full writeup, all the numbers, and every image are
in `report/Weon_Task4_Report.docx`.

Total cost across every experiment in this repo: about $3.80 of the $20
budget provided.

## How it works

Outside the region an edit was meant to touch, the output should be
identical to the input. That gives ground truth. Each step in an edit
script carries its own allowed rectangles; the analyzer masks those out and
measures everything else against the original: SSIM, PSNR, delta E color
drift, sharpness, near vs far field damage, and named "watch box" tracking
(logo, sign, face), per edit and cumulatively.

## Setup

Python 3.10 or newer. No virtual environment required.

    pip3 install -r requirements.txt
    cp .env.example .env
    # then paste your real keys into .env:
    #   OPENROUTER_API_KEY - used for every real, measured, report run
    #   GEMINI_API_KEY      - optional, only for --provider gemini rehearsals

If pip3 refuses with "externally-managed-environment":

    pip3 install --break-system-packages -r requirements.txt

Optional extras, install any time, columns fill in automatically once present:

    pip3 install lpips torch                  # adds cum_lpips column
    pip3 install insightface onnxruntime      # adds *_embed_sim column for the face watch box

## Generate the base image

Two ways to get data/inputs/base.png:

Via the API (scripted, cost-logged, repeatable), generates N candidates so
you can pick the best one:

    python3 src/generate_base.py --mock --n 3        # free rehearsal first
    python3 src/generate_base.py --provider gemini --n 3
    python3 src/generate_base.py --provider openrouter --n 3

Then open the printed folder, pick your favorite, and:

    cp data/runs/base_candidates_.../candidate_02.png data/inputs/base.png

Or manually: generate the image in the Gemini app using
prompts/base_image_prompt.txt, download it, and save it as data/inputs/base.png.

## Commands

Free sanity check, no API key needed, run this first:

    python3 tests/test_synthetic.py

Rehearse the whole pipeline for $0 before spending real credits (every mode
and flag works with --mock; outputs are stamped MOCK EDIT so they can never
be mistaken for real results):

    python3 src/run_chain.py --image data/inputs/base.png --script edit_scripts/chain_plain.json --mode full --mock --run-name rehearsal

Run one real chain (the three modes are full / crop / rebase):

    python3 src/run_chain.py --image data/inputs/base.png --script edit_scripts/chain_plain.json --mode full --run-name plain

Optional flags: --diff (diff compositing), --gate (verify-and-retry),
--best-of N (unconditional best-of-N), --size N (resolution management),
--provider {openrouter,gemini} (which real API to call; gemini uses your
free direct Gemini key, pass --model gemini-3.1-flash-image with it),
--model (override the default model for either provider).

Measure a finished run:

    python3 src/analyze_run.py --run data/runs/plain

Overlay several arms on one chart, the report's hero figure:

    python3 src/compare_runs.py --runs data/runs/plain data/runs/preserve data/runs/crop --labels plain preserve crop

Stack two arms row by row, step by step (the "same chain, with and without,
side by side" figure the brief asks for):

    python3 src/side_by_side.py --runs data/runs/plain data/runs/crop --labels plain crop

A full worked example, from a fresh chain to a finished comparison:

    python3 src/run_chain.py --image data/inputs/base.png --script edit_scripts/chain_plain.json --mode full --diff --run-name diff
    python3 src/analyze_run.py --run data/runs/diff
    python3 src/compare_runs.py --runs data/runs/plain data/runs/diff --labels plain diff

## Cost control

Every real API call (both providers) is logged with its exact cost and
latency to data/costs.csv, and the OpenRouter client refuses to spend past
MAX_BUDGET_USD in src/config.py.

One model, Seedream 4.5, is served through OpenRouter's newer dedicated
Image API rather than the standard chat-completions endpoint. or_client.py
detects this automatically (see IMAGES_API_MODELS) and routes accordingly,
so calling it works the same as any other model. Worth knowing if you point
this at a different third model later and hit an unexpected 404.

## Repository layout

    data/       every experiment's raw output: every step image, every
                manifest, every base-image candidate. Large and deep, meant
                for reproducing or auditing any single run in full.
    results/    a curated subset for quick browsing: the two hero charts,
                the chosen base image, the cost log, and one metrics.csv
                per run, without wading through data/.

## Files

    src/config.py          settings: paths, model, working size, budget cap
    src/or_client.py        OpenRouter client (the real, measured path)
    src/gemini_client.py    direct Gemini API client (free rehearsal path)
    src/generate_base.py    generates the base image itself via API (text-to-image)
    src/metrics.py          the ruler: masked SSIM, PSNR, delta E, sharpness
    src/compositing.py      crop-edit-composite and diff-compositing tools
    src/face_embed.py       optional ArcFace face-identity similarity
    src/run_chain.py        runs one edit chain (full/crop/rebase, flags)
    src/analyze_run.py      turns a run into metrics.csv, curve.png, strips
    src/compare_runs.py     overlays several runs on one chart
    src/side_by_side.py     stacks runs row by row for a step-by-step figure
    edit_scripts/           the instruction lists (plain, preserve, global)
    prompts/                the base image generation prompt
    tests/test_synthetic.py free end-to-end sanity check, no API key needed
    report/                 the finished report, Weon_Task4_Report.docx,
                             plus human_check.md with the human-judgment
                             check protocol
    results/                curated highlights, see Repository layout above
