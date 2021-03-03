TheDreamMachine - A fast, realtime CPU raytracer

DISCLAIMER :
------------

(C) 2013-2014 MadMann's Company

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

INSTALLATION :
--------------

You need to have SDL (1.2) and CMake installed on your system. To build the
project, just type:

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release
make -j

If you do not have SSE on your system, you can use `cmake -DCMAKE_BUILD_TYPE=Release -DUSE_OPTIMIZED_KERNELS=OFF`.
This will use intersection and shading routines in plain C (about ~2x slower).

EXAMPLES :
----------

To visualize OBJ files, you can use the viewer executable in the test directory.
To run the viewer, type:

cd build
./test/viewer <file.obj> <file.mtl>

(You can omit the MTL file, it will be found anyway)

Since culling is enabled, some scenes might show triangular holes in the geometry,
but you can disable this feature (which is useful to gain some speed) by setting the
`ENABLE_CULLING` CMake option to `OFF`.
