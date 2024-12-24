+++
date = '2024-12-09T20:55:15+08:00'
draft = false
title = 'NextWebConflict'
+++

记录一个遇到的名称冲突问题，使用one-api作为api管理的时候，如果要使用claude的模型，但是chatgptnextweb已经内置了claude的模型名称，会引发冲突，所以可以在custom models中加入：


+claude-3-5-sonnet-20241022@OpenAI=claude-3-5-sonnet-20241022


注意，还支持扩展用法，以下就是隐藏了所有模型，然后增加了claude, gemini，当然这里的gemini还是走的默认的google接口，而claude走的OpenAI的接口（也就是我们oneapi的接口）：

-all,+gemini-1.5-flash,+claude-3-5-sonnet-20241022@OpenAI=claude-3-5-sonnet-20241022
