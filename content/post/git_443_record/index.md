---
title: "Git_443_record"
date: "2025-01-17T13:02:02+08:00"
draft: false
---

# 记录一次clash Tun模式下git失灵（push pill）
> 参考https://ganzhixiong.com/p/b792e008/

也就是在clash内核在tun模式下出于安全问题禁用了22端口，所以

```shell
Connection closed by x.x.x.x port 22
```

```shell
# 测试22端口
ssh -T git@ssh.github.com
# 测试443端口
ssh -vT -p 443 git@ssh.github.com
```

修改.ssh/config就可以解决了
```yaml
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```