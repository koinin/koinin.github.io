---
title: "记录一次zsh配置流程"
date: "2024-11-03"
---
> 记录一次zsh配置流程（zsh + oh my zsh）

1. DD系统

> https://github.com/bin456789/reinstall

2. 安装zsh

```
apt install zsh -y
chsh -s /bin/zsh
# reboot

```

3. 安装 OMZ

```
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# 配置文件
~/.zshrc

# 插件
~/.oh-my-zsh/custom/plugins
```



