---
title: "Linux terminal 美化"
date: "2024-08-11"
---
通过.bashrc美化shell

```
PS1="\[\e[01;37m\]\t \[\e[01;32m\]\u\[\e[37m\]@\h \[\e[36m\]\w\[\e[m\]\\$ "

```

通过.vimrc美化vim

```
syntax on " 设置语法高亮 
set nocompatible
set nu " 设置行数显示 
set tabstop=4 " 设置tab缩进长度为4空格 
set autoindent " 设置自动缩进，适用所有类型文件 
set cindent " 针对C语言的自动缩进功能，在C语言的编程环境中，比autoindent更加精准 
set list lcs=tab:\\|\\ " 设置tab提示符号为 "|"，注意最后一个反斜杠后面要留有空格 
set cc=0 " 设置高亮的列，这里设置为0，代表关闭 set cursorline " 突出显示当前行 
set showmode 
set showcmd 
set mouse =a 
set t_Co=256 
set relativenumber 
set noerrorbells 
set vb t_vb=
set encoding=utf-8 
set termencoding=utf-8

set fileencodings=ucs-bom,utf-8,cp936,gb18030,big5,euc-jp,euc-kr 
set fileencoding=utf-8 
```
