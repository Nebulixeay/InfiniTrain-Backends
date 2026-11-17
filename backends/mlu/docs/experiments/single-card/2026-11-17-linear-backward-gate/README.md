# Linear/Matmul backward dtype regression

12 combinations and 44 numerical comparisons passed against independent double formulas (maximum absolute error 4.768e-7). Covers both Linear layouts and 2D/batched Matmul. This predates the Softmax fix.

Original evidence files are copied byte for byte. `manifest.json` records their SHA-256 hashes. Runtime commands and binary/source identities are in the linked metadata.

- [gate.argv](gate.argv)
- [gate.log](gate.log)
- [gate.rc](gate.rc)
- [gate.secs](gate.secs)
- [gate_summary.json](gate_summary.json)
- [manifest.json](manifest.json)
- [source_manifest.json](source_manifest.json)
