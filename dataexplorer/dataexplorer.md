# The Geometry of Truth: Dataexplorer

This page contains a plethora of interactive charts for exploring the data from the paper *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets* by Samuel Marks and Max Tegmark.

The charts on this page are all produced simply by performing PCA to visualize high-dimensional LLaMA-13B representations of our datasets.

## Basic datasets

Let's start off with our basic datasets, containing simple statements like "The city of Beijing is in China" (true) or "Fifty-eight is larger than sixty-one" (false). Mouse over the points below to see which statements they correspond to.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/basic_datasets.html" width="100%" height="500"></iframe>

We're not sure why the `smaller_than` dataset doesn't look as separated as the rest. But things look better when you go to 3D (below, right).

<div style="display: flex; justify-content: space-between;">
    <iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/cities_3d.html" width="48%" height="500"></iframe>
    <iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/smaller_than_3d.html" width="48%" height="500"></iframe>
</div>

Even with these simple plots, there's already lots to explore! For instance, for `larger_than`, we see two axes of variation: one separating the red and blue clouds, and one running parallel to the point clouds (pointing up and to the right). Can you figure out what this second axis of variation is? See below for the answer.

## Negations

Now let's introduce some more complicated logical structure to our statements. We'll start by negating statements by adding the word "not."

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/negations.html" width="100%" height="500"></iframe>

How do the visually apparent "truth directions" of the negated statements compare to the "truth directions" of the un-negated statements? Let's check:
<div style="display: flex; justify-content: space-between;">
    <iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/cities+neg_cities.html" width="48%" height="500"></iframe>
    <iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/sp_en_trans+neg_sp_en_trans.html" width="48%" height="500"></iframe>
</div>

Here we've done PCA on the two datsets together (after independently centering each; otherwise there would also be a translational displacement between them). You can toggle which datasets are shown by clicking on the plot legends.

What's going on here? There are many possibilities, but our best guess is something like this: LLaMA-13B has some direction <img src="https://latex.codecogs.com/gif.latex?\mathbf{\theta}_t"/> representing truth and another direction <img src="https://latex.codecogs.com/gif.latex?\mathbf{\theta}_f"/> representing some other feature which is *correlated* with truth on `cities` and *anti-correlated* with truth on `neg_cities`[^1]; this inconsistency in correlation would produce the observations above. See our paper for much more discussion on this topic.

[^1]: For instance, this other feature might encode the <a href="https://lre.baulab.info/">relation</a> "is in the country of". With unnegated statements like "The city of Beijing is in China," this feature correlates with being true; for negated statements like "The city of Paris is not in France" this feature *anti-correlates* with truth.

## Conjunctions and disjunctions

Now let's try some logical conjunctions and disjunctions. Mouse over the datapoints below to see what our conjunctive/disjunctive statements look like.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/conj_disj.html" width="100%" height="500"></iframe>

Does it look like the circled points form a bit of a separate cluster? We thought so, and indeed there's a pattern to those statements. See if you can figure out what it is (answer below).

## More diverse datasets

All of the datasets above were curated to contain statements which are *uncontroversial*, *unambiguous*, and *simple*. They are also not very diverse -- each dataset is formed from a single template.

In contrast, we'll now look at some *uncurated* datasets adapted <a href="https://arxiv.org/abs/2304.13734">from</a> <a href="https://github.com/Algorithmic-Alignment-Lab/CommonClaim/tree/main">other</a> <a href="https://rome.baulab.info/">sources</a>. Mouse over the plots below to get a sampling of the statements in these datasets.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/uncurated_datasets.html" width="100%" height="500"></iframe>

The first thing to note about these visualizations is that we don't see a clear separation into true and false clusters. This is because the datasets are more diverse. Recall that PCA identifies the most salient axes of variation for a dataset; in more diverse datasets, these axes are more likely to encode some truth-independent feature. For instance, the statements in `companies_true_false` are formed using three different templates, and the top 2 principal components mostly encode the difference between these templates. It's quite shocking that `common_claim_true_false`, consisting of statements as diverse as "Rabbits can partially digest memories" (false) or "Dolphins are capable of acts of impressive intelligence" (true) has as much true/false separation as it does.

If we want to see separation into true/false clusters, we can borrow one of the PCA bases identified from our cleaner datasets. For instance, here are our uncurated datasets visualized in the PCA basis extracted from our `cities` dataset.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/uncurated_datasets_cities_basis.html" width="100%" height="500"></iframe>

## Emergence over layers

So far, we've only been looking at layer 12. But by sweeping over the layers of LLaMA-13B, we can watch as the features which distinguish true statements from false ones emerge. Interestingly, there's a 4-layer offset between when `cities` separates and when `cities_cities_conj` (conjunctions of statements about cities) separates. This might be due to LLaMA-13B hierarchically building up concepts, with more composite concepts taking longer to emerge.

<img src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/emergence.gif" width=100% height="auto"/>

Here's an interactive version of the above with different datasets.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/emergence.html" width="100%" height="600"></iframe>

