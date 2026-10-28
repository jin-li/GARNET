# GARNET Testing and Verification Manual

This manual explains how to reproduce, extend and interpret GARNET's automated
checks. Use the [User Manual](../user/index.md) to run a calculation and the
[Software Methodology Manual](../methodology/index.md) to understand its equations
and implementation limits. This third manual connects those claims to executable
evidence and release checks.

Passing tests establish the tested contracts for the recorded code, inputs and
environment. They do not qualify a reactor, a nuclear-data library, a Serpent
estimator, an interpolated branch state or an untested GPU. Application validation
still needs independent reference calculations, uncertainty assessment and the
appropriate spatial, angular, energy and time refinement studies.

```{toctree}
:maxdepth: 2

running
coverage
maintaining
```

## Read the evidence at the right level

| Evidence | What it can establish | What remains to demonstrate |
| --- | --- | --- |
| Synthetic parser/contract test | Shape, orientation, units, provenance and rejection behavior for the fixture | Correctness and suitability of a real exported library |
| Analytic solution, conservation identity or independent dense pencil | A specified discrete operator or mathematical limit | Accuracy for a different geometry, closure or physical state |
| Published benchmark comparison | Agreement for its documented data, discretization and tolerance | General predictive accuracy or regulatory qualification |
| Mocked device API | Dispatch, transfer and iteration-budget contracts | Execution and numerical correctness on actual hardware |
| Actual device comparison | The tested backend/device path agrees with its stated references | Other hardware, driver versions, scales and performance |

The [coverage map](coverage.md) identifies representative executable references;
it is not a promise that every capability has equivalent evidence. Historical
reports in [verification and validation](../verification_validation.md) and
[development validation](../development/validation.md) retain their own revision
and environment scope. Reproduce the relevant tests before treating an older
report as evidence for a new change.
