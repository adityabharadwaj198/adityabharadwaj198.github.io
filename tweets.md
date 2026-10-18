---
layout: default
title: Tweets
permalink: /tweets/
---

# Tweets

Tweets I like.

{% for url in site.data.tweets %}
{% include tweet.html url=url %}
{% endfor %}
