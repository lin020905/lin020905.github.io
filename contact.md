---
title: /contact
layout: page
permalink: /contact/
---

{% assign form_endpoint = site.contact_form_endpoint | default: "" %}

<section data-lang="zh">
  <h1>联系我？</h1>
  <form action="{{ form_endpoint }}" method="post" aria-label="联系表单">
    <label class="terminal-field"><span>姓名：</span><input type="text" id="name-zh" name="name" autocomplete="name" required></label>
    <label class="terminal-field"><span>邮箱：</span><input type="email" id="email-zh" name="email" autocomplete="email" required></label>
    <label class="terminal-field terminal-field-message"><span>留言：</span><textarea rows="5" id="message-zh" name="message" required></textarea></label>
    <input type="hidden" name="_subject" value="网站联系表单的新消息">
    <input type="submit" value="[ 提交 ]"{% if form_endpoint == "" %} disabled aria-disabled="true"{% endif %}>
  </form>
  <br><br>
  {% if form_endpoint == "" %}
  <p>私密收件尚未启用：请先在 <code>_config.yml</code> 中填写 Formspree 表单端点。</p>
  {% else %}
  <p>提交内容会由 Formspree 私密转发到站长邮箱，不会公开显示。</p>
  {% endif %}
</section>

<section data-lang="en">
  <h1>Get in touch?</h1>
  <form action="{{ form_endpoint }}" method="post" aria-label="Contact form">
    <label class="terminal-field"><span>name:</span><input type="text" id="name-en" name="name" autocomplete="name" required></label>
    <label class="terminal-field"><span>email:</span><input type="email" id="email-en" name="email" autocomplete="email" required></label>
    <label class="terminal-field terminal-field-message"><span>message:</span><textarea rows="5" id="message-en" name="message" required></textarea></label>
    <input type="hidden" name="_subject" value="New message from the website contact form">
    <input type="submit" value="[ submit ]"{% if form_endpoint == "" %} disabled aria-disabled="true"{% endif %}>
  </form>
  <br><br>
  {% if form_endpoint == "" %}
  <p>Private delivery is not enabled yet: add the Formspree endpoint to <code>_config.yml</code> first.</p>
  {% else %}
  <p>Formspree privately forwards submissions to the site owner's inbox; messages are not published.</p>
  {% endif %}
</section>
