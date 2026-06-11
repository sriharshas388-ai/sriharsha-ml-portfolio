# Push to GitHub (one-time setup)

Repo is ready locally at:
`/Users/prashanthdomakonda/Documents/sriharsha-ml-portfolio`

## Steps

1. Create a new **public** repository on GitHub named: `sriharsha-ml-portfolio`
   - Do NOT initialise with README (already exists locally)

2. Run:

```bash
cd /Users/prashanthdomakonda/Documents/sriharsha-ml-portfolio
git remote add origin https://github.com/YOUR_USERNAME/sriharsha-ml-portfolio.git
git push -u origin main
```

3. Replace `YOUR_USERNAME` with your GitHub username (e.g. `sriharshasurannagari`).

4. Add this link to your PhD CV:
   `https://github.com/YOUR_USERNAME/sriharsha-ml-portfolio`

## Optional: generate outputs before pushing screenshots to README

```bash
# Project 1
cd health-risk-xai && pip install -r requirements.txt && python train_and_explain.py

# Project 2
cd ../healthcare-ml-benchmark && pip install -r requirements.txt && python benchmark_models.py

# Project 3
cd ../health-data-quality-kit && pip install -r requirements.txt
python -m health_dq.cli --input sample_data/patients_sample.csv --report outputs/dq_report.json
```

Outputs stay in `.gitignore` — add 1–2 screenshots to README manually if you want visuals.
