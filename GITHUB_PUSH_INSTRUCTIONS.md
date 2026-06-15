# Push portfolio to GitHub

Local commits are ready. GitHub login is required (token in keyring is expired).

## One-time setup

```bash
gh auth login -h github.com -p https -w
```

Choose: GitHub.com → HTTPS → Login with a web browser → authorize `repo` scope.

## Push

```bash
cd /Users/prashanthdomakonda/Documents/sriharsha-ml-portfolio
git push -u origin main
```

Repo: https://github.com/sriharshas388-ai/sriharsha-ml-portfolio
