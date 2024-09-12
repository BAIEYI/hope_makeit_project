在Linux或类Unix系统中用于构建C++项目的典型步骤：<br>
```
$ mkdir build
$ cd build
$ cmake -DPACKAGE_TUTORIALS=ON ..
$ make
```

`$ mkdir build`<br>
mkdir 是“make directory”的缩写，用于创建一个新的目录。<br>
build 是新创建的目录的名称。<br>
这行命令创建了一个名为 build 的新目录，通常用于存放构建过程中生成的文件。<br>
`$ cd build`<br>
cd 是“change directory”的缩写，用于更改当前工作目录。<br>
这行命令将当前工作目录更改为新创建的 build 目录。<br>
`$ cmake -DPACKAGE_TUTORIALS=ON ..`<br>
`cmake `是一个跨平台的安装（编译）工具，用于自动化软件的编译过程。<br>
`-DPACKAGE_TUTORIALS=ON `是传递给 cmake 的一个命令行参数，用于定义一个变量 PACKAGE_TUTORIALS 并将其值设置为 ON。这个变量通常用于在CMake配置过程中启用或禁用特定的构建选项。在这个例子中，它指示 cmake 在构建过程中包含教程。<br>
.. 是一个相对路径，表示当前目录的父目录。通常，CMakeLists.txt 文件位于项目的根目录，而 cmake 命令是在 build 目录中运行的，因此 .. 用于指定CMakeLists.txt文件的位置。<br>
`$ make`<br>
make 是一个用于构建项目的工具，它根据Makefile文件中的指令来编译和链接源代码。<br>
当 cmake 命令运行后，它会生成一个Makefile文件在 build 目录中。make 命令使用这个Makefile来构建项目。<br>
