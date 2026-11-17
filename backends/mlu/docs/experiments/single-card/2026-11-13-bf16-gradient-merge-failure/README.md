# BF16 gradient-merge failure before backward alignment

Forward and loss completed, backward stopped on mismatched FP32/BF16 gradient buffers (rc134). The log label param does not establish that the target was a model parameter; the stack identifies a BackwardPartial gradient merge.

Original evidence files are copied byte for byte. `manifest.json` records their SHA-256 hashes. Runtime commands and binary/source identities are in the linked metadata.

- [loss.csv](loss.csv)
- [manifest.json](manifest.json)
- [run.argv.txt](run.argv.txt)
- [train.log](train.log)
- [train.rc](train.rc)
