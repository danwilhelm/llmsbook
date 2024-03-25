# Chapter 2: Revealing the "True" Structure from Trained Feedforward Nets

In an LLM, each layer is comprised of two blocks. These are the attention block and the feedforward block, also known as a multi-layer perceptron (MLP).

In this chapter, we'll find a minimal solution to a circuit proposed in Steven Wolfram's ["What Is ChatGPT Doing ... and Why Does It Work?"](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/#writtings-c2c_above-58).

Training algorithms often have trouble with minimal-sized circuits, likely since the solution space is small. Indeed, Wolfram found that approximating a wave function using training required a hidden layer, but we'll see it's possible using a single layer. 

Along the way, we'll investigate how a one-layer network can approximate any function arbitrarily well. In doing so, we'll see that neural networks essentially approximate a "true function" that likely has a simpler representation than a neural net.

Understanding this, it is enticing to believe that large LLMs noisily approximate some "true" structure of language which has yet to be discovered.
