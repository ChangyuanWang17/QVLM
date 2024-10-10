# No kernel image available

This problem arises with the cuda version loaded by bitsandbytes is not supported by your GPU, or if you pytorch CUDA version mismatches. To solve this problem you need to debug ``$LD_LIBRARY_PATH``, ``$CUDA_HOME``, ``$PATH``. You can print these via ``echo $PATH``. You should look for multiple paths to different CUDA versions. This can include versions in your anaconda path, for example ``$HOME/anaconda3/lib``. You can check those versions via ``ls -l $HOME/anaconda3/lib/*cuda*`` or equivalent paths. Look at the CUDA versions of files in these paths. Does it match with ``nvidia-smi``?

If you are feeling lucky, you can also try to compile the library from source. This can be still problematic if your PATH variables have multiple cuda versions. As such, it is recommended to figure out path conflicts before you proceed with compilation.


__If you encounter any other error not listed here please create an issue. This will help resolve your problem and will help out others in the future.


# fatbinwrap

This error occurs if there is a mismatch between CUDA versions in the C++ library and the CUDA part. Make sure you have right CUDA in your $PATH and $LD_LIBRARY_PATH variable. In the conda base environment you can find the library under:
```bash
ls $CONDA_PREFIX/lib/*cudart*
```
Make sure this path is appended to the `LD_LIBRARY_PATH` so bnb can find the CUDA runtime environment library (cudart).

If this does not fix the issue, please try [compilation from source](compile_from_source.md) next.

If this does not work, please open an issue and paste the printed environment if you call `make` and the associated error when running bnb.
