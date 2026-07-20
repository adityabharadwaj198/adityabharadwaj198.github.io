---
layout: post
title: "Recent updates: search, retrieval, and learning ML properly"
date: 2026-07-20
categories: [notes]
tags: [machine-learning, search, retrieval, embeddings, learning]
excerpt: "What I've been working on at Uber Eats, the ML and retrieval ideas I've been learning, and what I want to explore next."
---

It's been a minute.

I joined Uber Eats in September, on the Search team. We focus on building the data pipelines that serve Uber's ranking and retrieval use cases—both offline training and online inferencing.

From December through April, I worked on pipelines that generate embeddings for store and item entities, then index them in a Lucene-based vector index for approximate nearest-neighbour search, specifically HNSW.

We also built more frequent delta pipelines for updates, additions, and deletes. These pipelines run inference to generate embeddings, publish them to a Kafka topic, and have a consumer ingest them into the vector index.

I enjoyed this work a lot. It gave me a much clearer view of what it takes to move from an embedding model to a production retrieval system.

## Things I learned

Some of the topics I encountered—and want to understand more deeply—were:

1. **Two-tower embedding models.** I understand the broad architecture, but I want to study them more deeply.
2. **Quantisation in vector indexes.**
3. **The search funnel:** retrieval-based candidate generation, first-pass ranking, second-pass ranking, hydration, and filtering.
4. **Data leakage in ranking models.**
5. **Online versus offline features.** Online inference can use real-time databases such as Cassandra or Redis, while offline feature generation and aggregation can use systems such as Hive.
6. **The feature lifecycle:** feature generation, aggregation, ingestion into an offline database, ingestion into an online database, and finally feature retrieval at prediction time.
7. **Semantic IDs and the cold-start problem.**
8. **Ways to generate Semantic IDs,** including residual quantisation, RQ-optimised product quantisation, and RQ-VAE.
9. **Sequence modelling:** representing a customer's sequence of actions to predict what they might do next.
10. **Generative retrieval and ranking using Semantic IDs.**
11. **Feature unification.** Putting the features needed for model inference into a single Cassandra or Redis table can reduce latency by bringing related data into the cache together and serving subsequent requests from it.

## Semantic IDs and generative retrieval

Semantic IDs are one of the areas I want to explore next. I plan to read [*Recommender Systems with Generative Retrieval*](https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf) and experiment with Semantic IDs on my own data.

The idea of replacing arbitrary item identifiers with IDs that encode semantic information is fascinating. It connects representation learning, quantisation, retrieval, and generation in a way that feels both theoretically interesting and practically useful.

## What I'm reading

I'm currently reading:

- *AI Engineering* by Chip Huyen
- *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst

I also have Chip Huyen's [guide to designing a machine learning system](https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html#project-setup-zlkQIG9) and [Machine Learning Interviews Book](https://huyenchip.com/ml-interviews-book/contents/0-about-the-questions.html) on my radar.

## Learning ML from the foundations

I've also been working through Andrej Karpathy's *Neural Networks: Zero to Hero* course. So far, I have:

- Built a bigram character-level language model from scratch
- Implemented a trigram character-level language model using sampling
- Implemented some neural networks from scratch

I intend to return to the course and eventually be able to build GPT from scratch. I'll probably share my solutions to the exercises as I go.

Going through the course made me realise that I never studied ML or AI properly at university. I want to fix that, so today I revisited some material from Andrew Ng's Deep Learning Specialization. I intend to implement the exercises from that course, advice that I got ~8 years back in college from my friends.

I also want to learn more about inference engineering. I can already think of a few interesting problems at Uber where that knowledge could be useful.

## System design implementations 

Last year when I was interviewing I implemented a few toy systems and they helped me remember things really well. Like consistent hashing - so this year I'm planning to take it forward by implementing LSM tree on my own. I'm procrastinating a lot on that because I can't figure out how skip lists work without having to look at someone's blog. 

## Elsewhere

I'm LeetCoding on the side because my stupid ass applied to a really nice company, got a call back, and would hate to interview without being prepared.

Outside work and study, I hit two reps at 95 kg (209 lb) on the bench. I've been enjoying training in the gym again after a long time.

I'm also guiding my little sister on ML / AI and having fun discussing topics with her that she's gotten more familiar with than me lol. I've also been funding her robotics pursuits. This makes me want to buy my own drone and program it. 

I'm also trying to practice delusional optimism a whole lot more these days. 