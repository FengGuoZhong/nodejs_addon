# nodejs addon

# Install

```bash
1、安装nodejs,本例使用的版本为 node-v4.4.3-x64.msi
2、在nodejs的npm安装目录下C:\Program Files\nodejs\node_modules\npm
   执行npm install node-gyp@3.4.0(请用管理员身份运行cmd,可以到C:\Windows\System32\cmd.exe)
3、安装Microsoft Visual Studio2015(2013)C++
4、安装Python-v2.7.10
5、配置环境变量Path = C:\Program Files\nodejs\node_modules\npm\bin\node-gyp-bin
6、进入示例目录（hello），执行node-gyp configure build，完成编译。