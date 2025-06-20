---
layout: default
title: Notes
permalink: /notes/
---

# Study Notes

Mostly my notes

<div class="post-list">
  {% for note in site.notes %}
    <article class="post-preview">
      <h2><a href="{{ note.url | relative_url }}">{{ note.title }}</a></h2>
      {% if note.excerpt %}
        <div class="post-excerpt">
          {{ note.excerpt }}
        </div>
      {% endif %}
      <a href="{{ note.url | relative_url }}" class="read-more">View Note →</a>
    </article>
  {% endfor %}
</div>

---

*This section is a work in progress. I'll be adding my study notes, tutorials, and learning resources here as I continue my learning journey.*

*Feel free to reach out if you have suggestions for topics you'd like me to cover!* 