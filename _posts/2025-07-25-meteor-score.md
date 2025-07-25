---
layout: post
title: "What the heck is the METEOR Score (and why I added it to Promptfoo)"
date: 2025-07-25
categories: [blog]
tags: [ai, evals, promptfoo, lexical-similarity]
excerpt: "A quick dive into METEOR, a smarter evaluation metric for LLM outputs, and a behind-the-scenes look at how I implemented it for Promptfoo."
---

I was recently going over Chip Huyen's AI engineering book (if you want a quick beginner friendly compilation you should check out the book). In Chapter 3 of the book titled Evaluation Methodology, she mentions different kinds of Lexical similarity metrics - which include BLEU, METEOR & ROUGE. It reminded me of the time when I implemented the METEOR metric for a eval + ai red teaming tool called Promptfoo. Let's dive into it. 

## First off — what is METEOR?

METEOR stands for Metric for Evaluation of Translation with Explicit ORdering. Cool acronym, what does it do? idea behind it is:

“How similar is the model's output to what I wanted it to say?”

It’s not just a simple string match. METEOR goes beyond that, it checks if the words match, whether they have the same root (stems), and even whether they’re synonyms. Oh, and it also factors in the word order.

## So why did I build this for Promptfoo?
Promptfoo is a powerful open-source framework to test and benchmark LLM outputs. It supports all kinds of evaluation methods — from latency to cost to BLEU (I implemented this too!), GLEU, ROUGE and even model-graded evaluations.

But there was a missing piece in the lexical similarity toolkit. What we needed was a metric that:

Works well for short, structured outputs (like translations or summaries)
Can account for minor word reordering
Supports multiple valid reference outputs
Understands synonyms and stems

*Oh, and they had an open issue where a Promptfoo user asked for being able to use the METEOR metric lol.* 

So I dived right into it and added METEOR support to Promptfoo.

## How METEOR works (with examples)

At a high level, METEOR compares a model-generated sentence with a reference by matching words in three stages:

1. Exact word match

2. Stemming match (e.g., "running" vs. "run")

3. Synonym match (e.g., "nice" vs. "beautiful", using WordNet)

Once matches are found, it calculates precision and recall, applies a penalty for how fragmented the matches are (i.e. word order differences), and outputs a final score between 0 and 1.

Here’s a quick example:

```
Reference:
The weather is beautiful today

Candidate outputs and expected scores (approximate):

The weather is beautiful today → 1.0 (exact match)

Today's weather is beautiful → ~0.85 (same words, reordered)

The weather is nice today → ~0.7 (synonym match on "nice")

It is sunny outside → ~0.3 (semantic drift, different words)
```
This kind of scoring is especially useful when there's more than one “acceptable” phrasing — something that comes up constantly in LLM outputs.

Unlike BLEU or Levenshtein, METEOR gives partial credit for “close enough” matches while still penalizing structural changes when they matter.

## How to use METEOR in Promptfoo

I wrote some documentation as part of my PR as well so I may as well point you directly to the documentation: https://www.promptfoo.dev/docs/configuration/expected-outputs/deterministic/#meteor

## When should you use METEOR?
If your tests are judging:

* Summaries
* Translations
* Short natural language answers

Anywhere there’s more than one “valid” way to say something
…it’s a great tool.

## Final Thoughts
Adding METEOR to Promptfoo was fun, both as a coding problem (chunking, synonyms, scoring!) and as a UX enhancement for LLM evaluators. I usually don't code in Typescript so it was a great exercise in that sense! Here's my PR if interested to check it out: https://github.com/promptfoo/promptfoo/pull/3776/files

If you want to go beyond BLEU and get a more nuanced picture of output quality, give it a spin. Your tests will thank you. And your models might pass more often. 