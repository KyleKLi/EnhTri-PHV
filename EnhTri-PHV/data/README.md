# Data

`base/` contains the LSTM-PHV interaction CSV supplied for Benchmark
construction. `raw/Benchmark/` contains the frozen positive, negative, all,
training, test, and sequence files used by the manuscript protocol.
`processed/Benchmark/` contains the deterministic tabular outputs consumed by
feature extraction, including the frozen protein-to-GO table.

The counts expected by the fixed protocol are:

- 22,383 positive pairs;
- 223,830 negative pairs;
- 196,970 training pairs;
- 49,243 independent-test pairs;
- 5,959 human and 1,072 viral accessions;
- 5,882 unique human sequences and 996 unique viral sequences in the positive
  interaction CSV.

Check `SHA256SUMS.txt` before use. Confirm the redistribution terms of the
source Benchmark before making a public release.
