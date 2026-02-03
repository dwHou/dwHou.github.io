# Bilingual Support Guide for index.html

This guide explains how to add new blog posts with bilingual (Chinese/English) support to your website.

## Overview

The bilingual system uses HTML5 `data-*` attributes to store both Chinese and English text for each element. JavaScript automatically switches the displayed language when the user clicks the language toggle button.

## Language Toggle Button

The language toggle button is positioned in the navigation bar after "ALGO". It shows:
- "EN" when the current language is Chinese (clicking switches to English)
- "中文" when the current language is English (clicking switches to Chinese)

## How to Add a New Post

### 1. Basic Post Structure

For a post **with an image**:

```html
<a href="posts/YOUR-POST-PATH">
<div class="post">
<div class="post-image">
<img src="images/post-covers/YOUR-IMAGE.png" alt="description">
</div>
<div class="post-content">
<h3 data-zh="中文标题" data-en="English Title">中文标题</h3>
<h4><i data-zh="中文副标题" data-en="English Subtitle">中文副标题</i></h4>
</div>
</div>
</a>
```

For a post **without an image** (text-only):

```html
<a href="posts/YOUR-POST-PATH">
<div class="post">
<div class="post-image">
<div class="post-placeholder"></div>
</div>
<div class="post-content">
<h3 data-zh="中文标题" data-en="English Title">中文标题</h3>
<h4><i data-zh="中文副标题" data-en="English Subtitle">中文副标题</i></h4>
</div>
</div>
</a>
```

### 2. Key Rules

1. **Always include both `data-zh` and `data-en` attributes** on elements you want to switch languages
2. **Set the initial text content** to the default language (Chinese in this case)
3. **The pattern is**: `data-zh="Chinese text" data-en="English text">Chinese text</element>`

### 3. Examples

#### Example 1: Simple Title and Subtitle
```html
<h3 data-zh="深度学习" data-en="Deep Learning">深度学习</h3>
<h4><i data-zh="学习理论进展" data-en="advances in learning theory">学习理论进展</i></h4>
```

#### Example 2: With Inline Links
```html
<h3>
    <a href="posts/link1.html">DDPM</a> &
    <a href="posts/link2.html">Score-based</a>
</h3>
<h4><i data-zh="扩散模型的原理与应用" data-en="The Principles and Applications of Diffusion Models">扩散模型的原理与应用</i></h4>
```

#### Example 3: Multiple Links in Subtitle
```html
<h3 data-zh="感受野计算器" data-en="Receptive Field Calculator">感受野计算器</h3>
<h4><i>
    <a href="posts/link1.html" data-zh="图交互" data-en="Visual">图交互</a> |
    <a href="posts/link2.html" data-zh="表交互" data-en="Table">表交互</a>
</i></h4>
```

### 4. Adding a New Section

To add a new section header:

```html
<h2 data-zh="新章节标题" data-en="New Section Title">新章节标题</h2>
```

### 5. Common Patterns

#### Computer Science / Technical Posts
- 中文标题: Technical term or Chinese description
- English subtitle: English technical description
```html
<h3 data-zh="AI智能体" data-en="AI Agents">AI智能体</h3>
<h4><i data-zh="智能代理系统" data-en="Intelligent Agent Systems">智能代理系统</i></h4>
```

#### Academic Papers
- Keep the title in English if it's a paper title
- Add Chinese/English descriptions in subtitle
```html
<h3 data-zh="Research Debt" data-en="Research Debt">Research Debt</h3>
<h4><i data-zh="转载自Distill" data-en="From Distill">转载自Distill</i></h4>
```

## Testing Your Changes

1. Open `index.html` in a web browser
2. Click the language toggle button (EN/中文) in the navigation bar
3. Verify that:
   - All post titles switch between Chinese and English
   - All post subtitles switch correctly
   - Section headers switch correctly
   - Navigation menu items switch correctly

## Technical Details

### How It Works

1. **Data Attributes**: Each bilingual element has `data-zh` and `data-en` attributes
2. **JavaScript Toggle**: The `toggleLanguage()` function switches between 'zh' and 'en'
3. **DOM Update**: The `updateLanguage()` function updates all elements with data attributes
4. **Button Update**: The button text changes to show which language will be shown next

### CSS Styling

The language toggle button uses these styles:
- Background: `#2AA7F7` (primary blue)
- Hover: `#1e88e5` (darker blue)
- Border radius: `20px` (rounded pill shape)
- Position: Last item in the navbar menu

## Troubleshooting

### Issue: Text doesn't switch
- Check that you've added both `data-zh` and `data-en` attributes
- Make sure the initial text content matches one of the data attributes

### Issue: Button doesn't appear
- Clear browser cache and reload
- Check browser console for JavaScript errors

### Issue: Formatting breaks when switching
- Avoid putting data attributes on container elements with complex children
- Use span elements to wrap text that needs translation

## Quick Reference

**Add a simple post:**
```html
<a href="YOUR-LINK">
<div class="post">
<div class="post-image"><img src="YOUR-IMAGE.png"></div>
<div class="post-content">
<h3 data-zh="中文标题" data-en="English Title">中文标题</h3>
<h4><i data-zh="中文副标题" data-en="English Subtitle">中文副标题</i></h4>
</div>
</div>
</a>
```

**Add a section:**
```html
<h2 data-zh="中文章节名" data-en="English Section">中文章节名</h2>
```

**Add a navigation item:**
```html
<li><a href="link.html"><span data-zh="中文" data-en="English">中文</span></a></li>
```

## File Locations

- Main file: `/Applications/Programming/code/dwHou.github.io/index.html`
- Reference implementation: `/Applications/Programming/code/dwHou.github.io/index_improve.html`
- This guide: `/Applications/Programming/code/dwHou.github.io/BILINGUAL_GUIDE.md`
