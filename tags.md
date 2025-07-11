---
layout: default
title: Tags
permalink: /tags/
---

<div class="tags-page">
  <h1>All Tags</h1>
  
  <div class="tags-list">
    {% assign tags = site.tags | sort %}
    {% for tag in tags %}
      {% assign tag_name = tag[0] %}
      {% assign posts = tag[1] %}
      <div class="tag-item">
        <a href="/tags/{{ tag_name | slugify }}/" class="tag-link">
          <span class="tag-name">{{ tag_name }}</span>
          <span class="tag-count">({{ posts.size }} post{% if posts.size != 1 %}s{% endif %})</span>
        </a>
      </div>
    {% endfor %}
  </div>
</div> 