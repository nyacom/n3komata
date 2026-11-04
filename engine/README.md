# Engine

This package builds the N3komata MoE engine from the llama.cpp upstream
revision `f5027fb21307eb264aed483242603d68c87b0fe3` and applies
`patches/llama.cpp/moe-correction.patch`.

The patch SHA-256 is
`0c535690bf1fa002132702438b436933f1cd4a8a0d72c2c7fb107f20d549b9fe`.
The upstream source is fetched by `materialize.py`; it is not vendored here.
llama.cpp source and notices retain their upstream license terms.
