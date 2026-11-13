# BitRound compatibility data

These fixtures establish a compatibility baseline from Numcodecs commit
`1f83681cf310ac9aac8041ed6ff4a5dcc42a7aad`
(`0.17.1.dev1+g1f83681cf`), generated with Python 3.13.5 and NumPy 2.5.4 on
little-endian Windows x86_64. They are not fixtures from an earlier release.

The input is `[1.15625, -1.15625, 2.6875, -2.6875]`, shape `(4,)`.
`encoded.00.dat` stores the encoded `<f4` input as `<i4`; `encoded.01.dat`
stores the encoded `<f8` input as `<i8`. Each `config.json` is the corresponding
codec's JSON configuration. The raw files follow the other codec fixtures'
`codec.NN/config.json` and `encoded.NN.dat` naming convention.

| Directory | keepbits | Expected decoded values (both float widths) |
| --- | --- | --- |
| `codec.00` | 2 | `[1.25, -1.25, 2.5, -2.5]` |
| `codec.01` | 3 | `[1.125, -1.125, 2.75, -2.75]` |

The expectations can be checked without Numcodecs. The positive inputs are
`37/32` and `43/16`, both exactly representable in float32 and float64. Keeping
`k` fraction bits gives spacing `2**-k` in `[1, 2)` and `2**(1-k)` in `[2, 4)`.
Nearest rounding therefore gives `5/4, 5/2` for `k=2` and `9/8, 11/4` for `k=3`;
the negative inputs give their negatives. None of these cases is a tie.
All expected values are also exactly representable, so no tolerance is needed.

The files were generated once at the stated baseline with:

```python
import json
from pathlib import Path

import numpy as np
from numcodecs import BitRound

data = [1.15625, -1.15625, 2.6875, -2.6875]
for j, keepbits in enumerate([2, 3]):
    codec = BitRound(keepbits=keepbits)
    directory = Path('fixture/bitround') / f'codec.{j:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(
        json.dumps(codec.get_config(), sort_keys=True, indent=4) + '\n'
    )
    for i, dtype in enumerate(['<f4', '<f8']):
        encoded = codec.encode(np.array(data, dtype=dtype))
        (directory / f'encoded.{i:02d}.dat').write_bytes(encoded.tobytes())
```

Generation was independently checked against exact rational rounding and
`struct.pack('<4f', *expected)` / `struct.pack('<4d', *expected)` for the values
above. The test only reads these files, round-trips the saved configuration
through JSON, and decodes with the registry-returned instance. It supplies the
integer dtype explicitly because BitRound does not store dtype or shape metadata.
The test checks reading this fixed data; it does not require future encoders to
produce identical bytes for arbitrary input or freeze all rounding behavior.
