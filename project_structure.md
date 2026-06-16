# SC-CLM Project Structure

```text
SC_CLM/
├── CLAUDE.md                     # Primary project operating guide
├── README.md
├── project_structure.md
├── .antigravity-rules/           # Coding and chemistry rules
├── .claude/agents/               # Custom reviewer/test/security/analysis agents
├── configs/                      # Config files
├── data/
│   ├── raw/                      # Raw NORMAN datasets
│   └── processed/                # train/val/test CSVs
├── src/
│   ├── data/                     # preprocessing, dataset logic
│   ├── model/
│   │   └── v5/                   # V5A/V5B training and inference code
│   ├── eval/                     # canonical metrics and benchmarks
│   └── routing/                  # experimental adaptive routing
├── logs/
│   ├── v5a/
│   ├── v5a1/
│   ├── v5b/
│   └── v5b1/
├── results/
│   └── checkpoints/
│       ├── v5a/
│       ├── v5a1_calib/
│       ├── v5b/
│       └── v5b1_calib/
├── scratch/                      # one-off scripts and audits
├── skills/                       # Local reusable chemistry skills
├── tools/                        # External/reference tooling
└── ref/                          # Archive and historical references
```
