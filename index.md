---
layout: default
title: Aditya Bharadwaj
description: Technology Brother
---

# Hi, I'm Aditya

Software engineer, heavy lifter, technology enthusiast. Curious about startups, history, space, geopolitics, climate, energy, physics and math. 

## About Me
Currently interested in vector stores, eval engines, and building my own toy systems. I loved space as a child and dreamt of being an astronaut (unironically). I hope to do space robotics one day. 

I spent ~4.25 years at Amazon building systems that power payment selection, payment processing and transaction history for India marketplace. Simplistically, if you've ever paid on Amazon, you have touched code that I wrote. I have spent almost ~0.75 years now at a early stage vector search engine company where we vectorise products using LLMs and store their vector embeddings in a vector store to enable search & recommendations usecases. 
When I'm not coding, you can find me exploring new music, working out, reading history or geopolitics.

## Connect With Me

- <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/gmail.svg" alt="Email" style="width: 20px; height: 20px; vertical-align: middle; margin-right: 6px;"> [Email](mailto:adityabharadwaj198@gmail.com)
- <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" alt="LinkedIn" style="width: 20px; height: 20px; vertical-align: middle; margin-right: 6px;"> [LinkedIn](https://www.linkedin.com/in/aditya-bharadwaj-28a91844/)
- <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" alt="GitHub" style="width: 20px; height: 20px; vertical-align: middle; margin-right: 6px;"> [GitHub](https://github.com/adityabharadwaj198/) - Check out my projects / OSS contributions.
- <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/readthedocs.svg" alt="Resume" style="width: 20px; height: 20px; vertical-align: middle; margin-right: 6px;"> [Resume](https://drive.google.com/file/d/1wMlSeLcq4WZmv5cWWQ_Es5ahJee-0BVd/view?usp=sharing)
- <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/twitter.svg" alt="Twitter" style="width: 20px; height: 20px; vertical-align: middle; margin-right: 6px;"> [Twitter](https://twitter.com/ad1tyabharadwaj) - shitposting
- <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/spotify.svg" alt="Spotify" style="width: 20px; height: 20px; vertical-align: middle; margin-right: 6px;"> [Spotify](https://open.spotify.com/user/adityabharadwaj198?si=f869c01e3ce34415) - check out my playlists. I put a lot of thought into making each one.

## Latest Blog Posts

<div class="latest-posts">
  {% assign all_content = site.posts | concat: site.notes %}
  {% assign sorted_content = all_content | sort: 'date' | reverse %}
  {% for item in sorted_content limit:3 %}
    <article class="post-card">
      <h3>
        <a href="{{ item.url | relative_url }}">
          {{ item.title }}
        </a>
      </h3>
      <div class="post-meta">
        <time datetime="{{ item.date | date_to_xmlschema }}">
          {{ item.date | date: "%B %-d, %Y" }}
        </time>
        {% if item.collection == "notes" %}
          <span class="note-label" style="margin-left: 10px; background: #f0f0f0; color: #007bff; border-radius: 8px; padding: 2px 8px; font-size: 0.8em;">Note</span>
        {% endif %}
      </div>
      {% if item.excerpt %}
        <div class="post-excerpt">
          {{ item.excerpt }}
        </div>
      {% endif %}
      {% if item.tags %}
        <div class="post-tags">
          {% for tag in item.tags %}
            <a href="/tags/{{ tag | slugify }}/" class="tag">{{ tag }}</a>
          {% endfor %}
        </div>
      {% endif %}
      {% if item.notes %}
        <div class="post-notes">
          <strong>Notes:</strong>
          <div class="notes-content">
            {{ item.notes }}
          </div>
        </div>
      {% endif %}
      <a href="{{ item.url | relative_url }}" class="read-more">Read More →</a>
    </article>
  {% endfor %}

  <div class="view-all-posts">
    <a href="{{ '/blog/' | relative_url }}" class="btn">View All Posts</a>
    <a href="{{ '/notes/' | relative_url }}" class="btn" style="margin-left: 10px;">View All Notes</a>
  </div>
</div>


---

*Last updated: {{ site.time | date: '%B %Y' }}*
