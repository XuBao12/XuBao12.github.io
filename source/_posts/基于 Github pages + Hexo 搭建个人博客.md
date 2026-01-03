---
title: 基于 Github pages + Hexo 搭建个人博客
---

## 为什么要搭建个人博客？

做学习笔记这件事情在我身上发生过很多次变化。最早追溯到大一开始用ipad进行无纸化学习，写下了几乎所有本科课程的笔记。大四时候读论文，为了让Windows端和Ipad端的Endnote论文笔记同步，开始用markdown记录一些随心所得。研究生之后接触到了神级的笔记软件Obsidian，加上用电脑敲代码的时间越来越多，经常随手就在Obsidian里记下了小知识点以及ideas。就这样，现在在我的本地电脑上已经形成了一个庞大的知识图谱体系，甚至于我有时能用Obsidian取代浏览器搜索一些遗忘的小知识点。

随着本地碎片化的知识体系越来越多，我生出了搭建个人博客的想法，权当有处地方记录我的读论文笔记。第一篇就记录一下搭建博客的经历和踩过的坑吧。

<!-- more -->

> 以下内容均在Windows上进行。

## 前置工作

1. 本地安装git以及nodejs。
2. 本地仓库部署Hexo环境。首先创建一个空文件夹，我命令为Blog，它就是博客站点的根目录。在根目录执行以下命令：

```bash
cd Blog #进入根目录
npm install -g hexo-cli #安装Hexo
hexo init #初始化Hexo框架
npm install #安装相关依赖包
hexo s #启动Hexo服务
```

没问题的话执行`hexo s`（等同`hexo server`）即可在本地浏览器访问`localhost:4000`看到默认站点了。



## 配置主题

我选择了Github上Hexo使用人数最多的Next主题。

将Next主题相关文件克隆到themes文件夹中`git clone git@github.com:next-theme/hexo-theme-next.git themes/next`。接着在根目录的配置文件`_config.yml`中修改参数`theme: next`，即可。

按需按照[官方文档](http://theme-next.iissnan.com/getting-started.html)配置个人信息、主题美化之类。



## 借助Github Pages部署

- Github新建仓库，仓库名称必须是`<username>.github.io`

- 执行命令`npm install hexo-deployer-git --save`以安装deploy-git

- 接着修改站点配置文件`_config.yml`的deploy部分为：

```yaml
deploy:
  type: git
  repo: https://github.com/YourgithubName/YourgithubName.github.io.git
  branch: master
```

- 执行命令即可在`<username>.github.io`链接上看到博客了。

```bash
hexo clean
hexo g
hexo d
```



> 其实插件deployer-git的作用就是把站点的public文件夹push到了远端仓库，可以手动设置。



## 写文章

有两种方式

1. 根目录执行`hexo new <title>`，在`/source/_posts/`下创建名为title的空markdown文件
2. 在`/source/_posts/`下新建md文件，然后在开头部分写下：

```
---
title: xxx
---
```





## Reference

- [开始使用 - NexT 使用文档 (iissnan.com)](http://theme-next.iissnan.com/getting-started.html)
- [2024年，如何使用 github pages + Hexo + Next 搭建个人博客 | PIPI's blog (mini-pi.github.io)](https://mini-pi.github.io/2024/02/28/how-to-make-blog-wedsite/#more)
- [Hexo+Next主题搭建个人博客+优化全过程（完整详细版） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/618864711)
- [Hexo + NexT 深度个性化配置 (2020.1 更新) | UPWNothing's Playground](https://upwnothing.github.io/zh-CN/44bf8def/#站点首页不显示文章全文)
