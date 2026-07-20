name: MRS Backfill (Manual)

on:
  workflow_dispatch:
    inputs:
      b20_pct:
        description: 'B20% from TradingView S5TW (e.g. 50.69)'
        required: false
        default: ''
      adl_tv:
        description: 'ADL from TradingView — auto ×1000 (e.g. 1827.69)'
        required: false
        default: ''
      zero_gamma:
        description: 'Zero Gamma SPX level (e.g. 7446.59)'
        required: false
        default: ''
      pc_ratio:
        description: 'PC Ratio — leave empty to auto-fetch'
        required: false
        default: ''

jobs:
  backfill:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - name: Configure git credentials
        run: git config --global url."https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/".insteadOf "https://github.com/"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run backfill
        env:
          B20_PCT:    ${{ inputs.b20_pct }}
          ADL_TV:     ${{ inputs.adl_tv }}
          ZERO_GAMMA: ${{ inputs.zero_gamma }}
          PC_RATIO:   ${{ inputs.pc_ratio }}
        run: python backfill.py
      - name: Commit updated history
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add mrs_history.csv
          git diff --staged --quiet || git commit -m "backfill: fill missing sessions $(date +%Y-%m-%d)"
          git push
