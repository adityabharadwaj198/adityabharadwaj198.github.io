---
layout: default
title: Aditya Bharadwaj
description: Technology Brother
---

<!-- Running Pikachu -->
<img src="{{ '/assets/img/pikachu.gif' | relative_url }}" alt="Running Pikachu" class="running-pikachu">

<header class="home-intro">
  <h1>Hi, I'm Aditya</h1>

  <p>Software engineer, DYEL bro, technology enthusiast. Curious about startups, history, space, geopolitics, climate, energy, physics and math.</p>

  <p>Currently interested in vector stores, eval engines, and building my own toy systems. I loved space as a child and dreamt of being an astronaut (unironically). I hope to do space robotics one day.</p>

  <p>I spent ~4.25 years at Amazon building systems that power payment selection, payment processing and transaction history for India marketplace. Simplistically, if you've ever paid on Amazon, you have touched code that I wrote. I have spent almost ~0.75 years now at a early stage vector search engine company where we vectorise products using embedding models and store their vector embeddings in a vector store to enable search &amp; recommendations usecases.</p>

  <p>When I'm not coding, you can find me exploring new music, working out, reading history or geopolitics.</p>
</header>

## Connect

<ul class="connect-links">
  <li><a href="mailto:adityabharadwaj198@gmail.com">email</a></li>
  <li><a href="https://www.linkedin.com/in/aditya-bharadwaj-28a91844/" target="_blank" rel="noopener">linkedin</a></li>
  <li><a href="https://github.com/adityabharadwaj198/" target="_blank" rel="noopener">github</a></li>
  <li><a href="https://drive.google.com/file/d/1wMlSeLcq4WZmv5cWWQ_Es5ahJee-0BVd/view?usp=sharing" target="_blank" rel="noopener">résumé</a></li>
  <li><a href="https://twitter.com/ad1tyabharadwaj" target="_blank" rel="noopener">x / twitter</a></li>
  <li><a class="spotify-link" href="https://open.spotify.com/user/adityabharadwaj198" target="_blank" rel="noopener">spotify</a></li>
</ul>

## Latest writing

<div class="latest-posts">
  {% assign all_content = site.posts | concat: site.notes %}
  {% assign sorted_content = all_content | sort: 'date' | reverse %}
  {% for item in sorted_content limit:3 %}
    <article class="post-card">
      <h3><a href="{{ item.url | relative_url }}">{{ item.title }}</a></h3>
      <div class="post-meta">
        <time datetime="{{ item.date | date_to_xmlschema }}">{{ item.date | date: "%b %Y" }}</time>
        {% if item.collection == "notes" %}
          <span class="note-label">Note</span>
        {% endif %}
      </div>
      {% if item.excerpt %}
        <div class="post-excerpt">{{ item.excerpt }}</div>
      {% endif %}
      {% if item.tags %}
        <div class="post-tags">
          {% for tag in item.tags %}
            <a href="/tags/{{ tag | slugify }}/" class="tag">{{ tag }}</a>
          {% endfor %}
        </div>
      {% endif %}
      <a href="{{ item.url | relative_url }}" class="read-more">Read more →</a>
    </article>
  {% endfor %}

  <div class="view-all-posts">
    <a href="{{ '/blog/' | relative_url }}" class="btn">All writing →</a>
    <a href="{{ '/notes/' | relative_url }}" class="btn">All notes →</a>
  </div>
</div>
