---
title: "Shadowrocket UDP 设置"
date: "2024-03-21"
draft: true
---

Shadowrocekt很好用，但是网上基本没看到他的文档，大家都把他当作ios代理入门软件，但实际上，Sr也有很多没写出来的特点。

以UDP Relay为例，如果关闭UDP转发，不管你是不是UoT，此时UDP都是走的直连，也就是会出现Webrtc泄露之类的问题，开启UDP转发之后，此时Webrtc会出现没有泄露但是也检测不到代理服务器的状态。只有开启UDP转发，并且打开UoT才可以使得Webrtc指向代理服务器。

同理，Clash (mihomo)也是有这个特性的，开启udp, UoT 才是完整的UDP配置
