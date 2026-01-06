mkdir build
cd build
deactivate # deactivate venv if loaded
export PYTHON_EXECUTABLE=/Applications/ParaView-6.0.1.app/Contents/bin/pvpython
cmake \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  ..
make -j

