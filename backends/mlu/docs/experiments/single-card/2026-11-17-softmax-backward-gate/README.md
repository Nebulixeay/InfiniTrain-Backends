# Softmax backward reproduction and regression

Before correction: FP32 gradient read as BF16, wrong output dtype and maximum absolute error about 5.849e28 (expected failure). After promotion: 9 dtype/shape cases pass, maximum absolute error 7.451e-8, including [4,12,64,64].

Original evidence files are copied byte for byte. `manifest.json` records their SHA-256 hashes. Runtime commands and binary/source identities are in the linked metadata.

- [baseline-argv.txt](baseline-argv.txt)
- [baseline-gate.log](baseline-gate.log)
- [baseline-gate.rc](baseline-gate.rc)
- [baseline-identity.sha256](baseline-identity.sha256)
- [baseline-seconds.txt](baseline-seconds.txt)
- [fixed-argv.txt](fixed-argv.txt)
- [fixed-gate.log](fixed-gate.log)
- [fixed-gate.rc](fixed-gate.rc)
- [fixed-identity.sha256](fixed-identity.sha256)
- [fixed-seconds.txt](fixed-seconds.txt)
- [manifest.json](manifest.json)
