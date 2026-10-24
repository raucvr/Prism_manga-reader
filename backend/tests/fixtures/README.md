# Test fixtures

The extraction, evidence and job tests use four open-access arXiv papers (pinned versions). They are not committed;
download them once:

```bash
cd backend/tests/fixtures
curl -L -o attention_is_all_you_need.pdf https://arxiv.org/pdf/1706.03762v7
curl -L -o chain_of_thought.pdf https://arxiv.org/pdf/2201.11903v6
curl -L -o resnet.pdf https://arxiv.org/pdf/1512.03385v1
curl -L -o bert.pdf https://arxiv.org/pdf/1810.04805v2
```

Tests that need a PDF are skipped automatically when the files are missing.
