# Migration from NDgpu to GARNET

The upstream project is now [poncetma/GARNET](https://github.com/poncetma/GARNET).
This fork is [jin-li/GARNET](https://github.com/jin-li/GARNET). The upstream
rename, license and public diffusion API changes are merged with the fork's
existing NEM, Serpent, adjoint kinetics, finite-volume and AMD extensions.

## Python and environments

The distribution and import package are `garnet`. Replace `import ndgpu` and
`from ndgpu...` with `import garnet` and `from garnet...`, then reinstall the
project (`python -m pip install -e '.[test]'`, or the documented locked uv
workflow). An existing editable `ndgpu` installation is not the renamed package.
Do not install an unrelated package with a similar name from a registry.
The fused-kernel switch is now `GARNET_FUSED`, replacing `NDGPU_FUSED`.
Manual PDF filenames use `garnet-`; the Colab upload is `dist/garnet-src.zip`.
The fork's CUDA, ROCm and documentation extras remain available.

Upstream now supplies the GNU GPL v3 license in `LICENSE`; package metadata
follows that file. Historical commits retain their recorded license metadata.

## Git and research provenance

Update existing clone URLs with:

```sh
git remote set-url origin https://github.com/jin-li/GARNET.git
git remote set-url upstream https://github.com/poncetma/GARNET.git
```

The integration preserves ancestry rather than rewriting commits. Maintained
`main`, `dev` and active feature branches receive the migration. Archived
experimental branches and original benchmark revisions retain their exact
history, as described in the [integration inventory](integration_inventory.md).
They can still be checked out to reproduce previously reported calculations.
A repository rename does not require renaming the local checkout directory.

Historical result keys, commit hashes and numerical inputs must not be changed
merely to replace the project name. For a frozen study, use its recorded
software revision. Updated study runners may explicitly select the historical
`ndgpu` package for those old checkouts, while current runs select `garnet`.
Private study repositories and manuscript submodules remain separate from the
public software wheel, source archive and source distribution.

On feature branches, new branch-library and regional-XS manifests write
`garnet.branch-xs.v1` and `garnet.regional-xs.v1`. Readers also accept the
corresponding historical `ndgpu.*.v1` schema names without changing stored
scientific values. Newly generated thermal-star/import provenance uses the
`garnet` namespace. The optional fitted-mesh dependency remains available as
`garnet[mesh]`; GPU test opt-in uses `GARNET_TEST_GPU=1`.
