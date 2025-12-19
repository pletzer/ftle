mkdir build
cd build
deactivate # deactivate venv if loaded
export PYTHON_EXECUTABLE=/Applications/ParaView-6.0.1.app/Contents/bin/pvpython
cmake ..
make -j

