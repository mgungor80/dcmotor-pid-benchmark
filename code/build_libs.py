"""Compile the C simulators into shared libraries next to this file (gcc or cc required).
Linux/macOS: python build_libs.py.  Windows: use MinGW-w64 gcc or WSL (the Python modules load lib*.so;
on Windows rename the outputs or adapt the CDLL calls in motor.py and sim4.py)."""
import os, subprocess, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
cc = os.environ.get('CC') or shutil.which('gcc') or shutil.which('cc')
if not cc: sys.exit('No C compiler found (set CC).')
for src in ('dcsim2', 'dcsim3', 'dcsim4'):
    cmd = [cc, '-O2', '-shared', '-fPIC', '-o', os.path.join(HERE, f'lib{src}.so'), os.path.join(HERE, f'{src}.c'), '-lm']
    print(' '.join(cmd)); subprocess.check_call(cmd)
print('ok')
