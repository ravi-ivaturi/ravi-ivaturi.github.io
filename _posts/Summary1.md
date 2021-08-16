---
layout: post
title: test1
date: 2021-08-16
ext-js: "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
---

## Summary

   Research Paper : Statistics and Causal Inference
  Author: Paul W. Holland

## Notation


$$U : set\;of\;units\\
K : (t, c)\\
Y : real\;valued\;function\;on\;U*K\\
S : mapping\;from\; U\;to\; ~K\\
A : set\;of\;attributes\;of\;u\;in\;U$$


## Key Ideas

### Associational Inference vs Rubin’ Causal Inference
Associational Inference model (A)  is simply descriptive in character and specifies the joint distribution of  Y and A over U whereas Rubin’s Causal Inference model (A) computes the conditional expectations. Further, the model considers attributes as having a different ontological status than causes.
Specifically, 



$$A : (U, Y, A)\\
Pr(Y = y, A = a)\\
R = (U, K, Y, S)\\
E(Y_t|S = t) , E(Y_c|S = c)$$



### The fundamental problem of causal inference 
It is impossible to observe both  
\[Yt\ and \Yc  over U\]

### How Rubin’s model addresses the the fundamental problem of causal inference_  Although we can only observe
$$E(Y_t|S = t) and E(Y_c|S = c)$$
the independence assumption , usually justified through experimental design allows us to conclude 

$$E(Y_t|S = t) = E(Y_t)\\
E(Y_c|S = c) = E (Y_c)$$

### Reconciling Rubin’s model with Hume’s views
David Hume was skeptical of even the possibility of inferring causality and specified three criteria : some version of ‘ No action at a distance’ (side-note: Hume would have been turned off by  quantum entanglement),  ‘causes precede effects’ and constant conjunction (i.e. cause and effect always exist as a pair like the magnetic poles). It’s easy to see that Rubin’s mode almost trivially satisfies the first two criteria (by  design). The third criteria, is of course harder to meet in a demonstrable way because of measurement error. 
### Jon Stuart’s Mill’s methods
   Mill’s methods, such as ‘Method of Difference’, ‘ Method of Residues’  have what we can recognize as an Econometric flavor and is reasonably aligned with Rubin’s. He too, however does not recognize a categorical difference between attributes and causes.
### Fisher and the philosophical question of ‘what’s a cause ?’
Fisher has been criticized for his views on the link between lung cancer and smoking - however the criticism did not seriously engage with the philosophical question of what counts a cause. Rubin’s model offers a recipe :  anything that could be considered as ‘ in- principle treatment in an experiment’
It’s worth noting both ‘in-principle’  and ‘experiment’ are used in a broad, if consistent way and not limited to an artificial setting.

## Personal Reflection
_Bayesian attitude_.  Ex-ante, all claims don’t have the same epistemic status. For example, the alleged detection of faster than light neutrinos in LHC is not on the same footing ex-ante  as the detection of gravitational waves. The former violates one of the most fundamental theories in modern physics whereas the latter is a confirmation of one of the predictions of the same theory. While this attitude is implicit in scientific research, it may be helpful to spell it out in Bayesian terms ; i.e. extraordinary claims need extra-ordinary evidence. In terms of Rubin’s model , the uncertainty around our inferential conclusions should be  quantified in bayesian  terms.


