---
layout: default
title: Blog
permalink: /blog/
---

<div class="blog-index">
  <h1>Blog Posts</h1>
  
  <div class="post-list">
    {% for post in site.posts %}
      <article class="post-preview">
        <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
        <div class="post-meta">
          <time datetime="{{ post.date | date_to_xmlschema }}">
            {{ post.date | date: "%B %-d, %Y" }}
          </time>
          {% if post.categories.size > 0 %}
            • Categories: {{ post.categories | join: ", " }}
          {% endif %}
        </div>
        {% if post.excerpt %}
          <div class="post-excerpt">
            {{ post.excerpt }}
          </div>
        {% endif %}
        <a href="{{ post.url | relative_url }}" class="read-more">Read More →</a>
      </article>
    {% endfor %}
  </div>
</div> 