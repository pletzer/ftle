# Loading the Paraview plugin

This directory contains plugins that can be loaded into Paraview.

## Building the plugin

Plugins call C++ code that must first be compiled. You must have a C++ compiler and CMake installed, in addition to Paraview. The steps were tested on Mac OS X with Paraview 6.0.1.

### On MAC OS X

Select the Python to be that of Paraview
```bash
export PYTHON_EXECUTABLE=/Applications/ParaView-6.0.1.app/Contents/bin/pvpython
```
Then, in the `paraview_plugin` directory
```bash
mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
 ..
make
```
Now you should have a shared library `ftlecpp.cpython-312-darwin.so` (the name will change depending on 
platform). Copy this file to the `paraview_plugin` directory.
```bash
cp ftlecpp.cpython-312-darwin.so ..
```

### On Linux

TO WRITE

## How to load a plugin

Start Parview. Under `Tools` -> `Manage plugins...`, then press `Load New`, navigate to the directory where your plugin resides. Select the plugin (PalmFtleSource) and press `OK`.  
Wait a few seconds, giving Paraview the time to load the plugin. Close the `Plugin Manager` window.

## How to invoke the plugin

For the `PALM FTLE Source` plugin, go to `Sources` and select `PALM FTLE Source` under the `Alphabetical` menu. Select the Palm file in the menu. You might have to press `Apply` to see the plugin.

## Volume rendering

The FTLE field is cell centred and therefore selecting `Volume` will not work. Additionally, volume rendering requires image data (i.e. uniform grid data) whereas the data are stored on a rectilinear grid. However, you can add a `Cell to Point Data` connecting to a `Resample to Image` filter, then use `Volume` to see the interior.  

