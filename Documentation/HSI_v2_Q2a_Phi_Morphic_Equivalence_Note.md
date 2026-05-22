# Q2a Analytic Note: Is Variant B a Morfic Substitution?

Status: working analytic note, not a Decision Log entry.

Date: 2026-05-19

## Question

Hostile-review Q2a asks whether the golden-ratio fingerprints reported for
Variant B can be derived analytically from a hidden morfic/Fibonacci-like
substitution, rather than from the full HSI Level 0 dynamics.

The concrete test is:

1. Is the canonical B collapse equivalent, under a suitable representation, to
   a fixed morfic substitution?
2. If yes, can the ratios `LZ -> 1/phi` and `cbar -> phi + 1` be derived from
   that substitution without running the HSI pipeline?

## Code anchor

The current executable semantics are in:

- `level0/generator.py::_simplify_base`
- `level0/generator.py::_collapse_inside_parentheses_local`
- `level0/generator.py::_collapse_global_ignore_parentheses`
- the `variant == "B"` branch of the main generation loop

The important post-Dec-2025 correction is that opposing pairs degrade to `0`
(No-Res), not to the empty string:

```text
01 -> 0
10 -> 0
0+ -> 0
1+ -> 1
```

iterated until stable.

## Local simplifier

The local simplifier induced by `_simplify_base` can be summarized as:

- a non-empty all-`1` word maps to `1`;
- every other non-empty observable word maps to `0`.

Equivalently, for a binary word `w`, the simplifier is an AND-like collapse:

```text
s(w) = 1 iff every observable symbol in w is 1
s(w) = 0 otherwise
```

This already differs from a Fibonacci-type substitution. It is a many-to-one
collapse, not a length-expanding letter morphism.

## Global B recurrence

At each iteration B does not apply a fixed substitution to each symbol of the
previous observable string. Instead it:

1. appends the current state to the accumulated historical trace `Acc`;
2. builds a framed decay object `(Acc)1`;
3. repeatedly collapses innermost parenthesized regions with the local
   simplifier, appending intermediate framed states to `Acc`;
4. applies a final global simplification if needed.

Thus the emitted observable stream is not `mu^n(a)` for a fixed morphism
`mu : Alphabet -> Alphabet*`. It is a history-dependent transduction whose next
emission depends on the full accumulated trace and on the dynamically produced
sequence of intermediate collapse states.

## Why this is not a fixed morphic substitution

A fixed morphic substitution has three relevant properties:

1. Each symbol has a context-independent image.
2. The next word is obtained by concatenating those symbol images.
3. Growth and letter frequencies are controlled by a finite substitution
   matrix.

Canonical B violates all three:

1. Collapse of a symbol depends on its parenthesized context and on the full
   accumulated trace in which it occurs.
2. The next emitted material includes intermediate collapse states, not only a
   direct image of the previous observable word.
3. The effective "production matrix" is not fixed: the amount and content of
   emitted material changes with the nested structure of `Acc` and with the
   number of inside-out collapse passes required by the current frame.

This does not prove that no higher-level morphic representation can ever be
constructed for some compressed observable of B. It does rule out the simple
hostile-review reading: B is not presently just a Fibonacci morphism in
disguise under the executable Level 0 semantics.

## Ratio derivability

Because there is no fixed substitution matrix for B under the current
implementation, the standard analytic route to golden-ratio limits is not
available. In a Fibonacci morphism, phi appears as the Perron-Frobenius
eigenvalue of the substitution matrix. For B, there is no fixed two-letter
matrix whose eigenvalue directly yields:

- `LZ -> 1/phi`;
- `cbar -> phi + 1`.

Those ratios may still be real empirical fingerprints of the generated stream,
but they are not derived here from a fixed morphic substitution. Any future
analytic derivation would need to model the history-dependent accumulation and
inside-out collapse process itself, not merely identify a hidden Fibonacci
substitution.

## Dictamen

Q2a verdict:

```text
not-morphic-under-current-executable-semantics
```

Interpretation:

- The phi ratios should not be presented as analytically forced by a known
  Fibonacci substitution.
- They should also not be over-promoted as independent evidence by themselves.
- The safe reading remains: phi is a numerical side fingerprint inside a wider
  empirical signature, and Q2b is still needed to test whether the quintuple
  signature survives a structurally different collapse rule.

## Caveat

The legacy document `Documentation/phi_emergence_mechanism.md` contains useful
historical context but uses older deletion-style language in places. The current
analysis follows the executable post-Dec-2025 semantics in `level0/generator.py`,
where degradation returns `0`, never the empty string.
