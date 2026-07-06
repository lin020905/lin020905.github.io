# lin020905.github.io

lin 的 GitHub Pages 个人网站。站点使用 Jekyll 风格结构组织项目和随笔，首页会自动读取 Markdown collection 内容。

## Structure

- `index.html` - Jekyll 首页模板，渲染项目和随笔索引
- `_layouts/` - 页面、项目、随笔的公共布局
- `_includes/` - 头部、底部和页面 head
- `_projects/` - 项目 Markdown collection
- `_notes/` - 随笔 Markdown collection
- `styles.css` - Gravity-inspired 极简视觉样式
- `script.js` - 中英切换和轻量滚动出现效果

## Add A Project

在 `_projects/` 里新增一个 `.md` 文件：

```yaml
---
title: 项目中文标题
title_en: Project English Title
type: Web / App
summary: 中文项目摘要。
summary_en: English project summary.
link: https://github.com/lin020905
order: 4
---
```

正文可以继续写项目背景、技术栈、截图说明和后续计划。

## Add A Note

在 `_notes/` 里新增一个 `.md` 文件：

```yaml
---
title: 随笔中文标题
title_en: English Note Title
date: 2026-07-07
tag: note
summary: 中文摘要。
summary_en: English summary.
---
```

## Local Preview

如果本机安装了 Ruby 和 Bundler：

```bash
bundle install
bundle exec jekyll serve
```

GitHub Pages 会在推送到 `main` 后自动构建站点。
