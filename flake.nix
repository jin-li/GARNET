{
  description = "NDgpu development environment (Nix Python and uv)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/b6c8664de9b6cc07fe5666a29f91884ba81197c4";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      # Keep the CPU/CUDA shell small. ROCm's split Nix outputs need a common
      # prefix for CuPy's header discovery and HIPRTC compilation.
      rocm = pkgs.rocmPackages;
      rocmPaths = with rocm; [
        clr hipblas hipsparse rocsparse rocrand hiprand rocsolver
        rocfft hipfft hipcub rocprim rocthrust rccl roctracer
        rocm-device-libs rocm-runtime rocm-comgr
      ];
      rocmHome = pkgs.symlinkJoin {
        name = "ndgpu-rocm-${rocm.clr.version}";
        paths = rocmPaths;
      };
    in {
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [ pkgs.python313 pkgs.uv ];
          shellHook = ''
            export UV_PYTHON="${pkgs.python313}/bin/python3.13"
            export UV_PYTHON_DOWNLOADS=never
            export UV_CACHE_DIR="$PWD/.cache/uv"
            export PYTHONNOUSERSITE=1
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export CUPY_CACHE_DIR="$PWD/.cache/cupy"
          '';
        };
        docs = pkgs.mkShell {
          inputsFrom = [ self.devShells.${system}.default ];
          packages = [ pkgs.gnumake (pkgs.texliveSmall.withPackages (ps: with ps; [
            latexmk lm capt-of ellipse framed needspace pict2e tabulary titlesec
            varwidth wrapfig
          ])) ];
          shellHook = ''
            export TEXMFVAR="$PWD/.cache/texmf-var"
            export TEXMFCONFIG="$PWD/.cache/texmf-config"
          '';
        };
        rocm = pkgs.mkShell {
          inputsFrom = [ self.devShells.${system}.default ];
          packages = [ rocmHome rocm.rocminfo rocm.hipcc ];
          shellHook = ''
            export ROCM_HOME="${rocmHome}"
            export ROCM_PATH="$ROCM_HOME"
            export HIP_PATH="${rocm.clr}"
            export HIP_PLATFORM=amd
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath rocmPaths}:''${LD_LIBRARY_PATH}"
          '';
        };
      };
    };
}
