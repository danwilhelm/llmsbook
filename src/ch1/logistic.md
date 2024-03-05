# Logistic Regression Learnings

It is insightful to consider how this chapter's transformer compares to a classical machine learning technique -- Logistic Regression. In this section, we'll apply logistic regression to the same problem, learn an important strategy for achieving a higher score, then apply the strategy to our handcrafted unembeddings.

> The key insight? Currently, our unembeddings are all positive letter frequencies that sum to 1. This allows us to support likely classes, but it doesn't allow us to penalize classes we know it's _not_. If we observe letters that should be rare for a given rotation class, by making that unembed _negative_ we can repel from that class and deliver a higher score.

We showed earlier that we could combine attention's value and output projections into a single effective transformation. In our current model, we have yet another linear projection immediately following these! So, this can be combined together to make the entire model effectively a single linear projection with bias!


