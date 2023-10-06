# The Geometry of Truth: Dataexplorer

This page contains a plethora of interactive charts for exploring the data from the paper *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets* by Samuel Marks and Max Tegmark.

The charts on this page are all produced simply by performing PCA to visualize high-dimensional LLaMA-13B representations of our datasets.

## Basic datasets

Let's start off with our basic datasets, containing simple statements like "The city of Beijing is in China" (true) or "Fifty-eight is larger than sixty-one" (false). You can see which statements the points shown correspond to by mousing over.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/basic_datasets.html" width="100%" height="500"></iframe>

We're not sure why the `smaller_than` dataset doesn't look as separated as the rest. But things look better when you go to 3D (below, right).

<div style="display: flex; justify-content: space-between;">
    <iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/cities_3d.html" width="48%" height="500"></iframe>
    <iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/smaller_than_3d.html" width="48%" height="500"></iframe>
</div>

Even with these simple plots, there's already lots to explore! For instance, for `larger_than`, can you figure out how the statements vary as you run move along the point clouds (up and to the right)? See below for the answer.

## Negations, conjunctions, and disjunctions

Now let's introduce some more complicated logical structure to our statements. We'll start by negating statements by adding the word "not."

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/negations.html" width="100%" height="500"></iframe>

How do the visually apparent "truth directions" of the negated statements compare to the "truth directions" of the un-negated statements? Let's check:
<div style="display: flex; justify-content: space-between;">
    <iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/cities+neg_cities.html" width="48%" height="500"></iframe>
    <iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/sp_en_trans+neg_sp_en_trans.html" width="48%" height="500"></iframe>
</div>

Here we've done PCA on the two datsets together (after independently centering each; otherwise there would also be a translational displacement between them). You can toggle which datasets are shown by clicking on the plot legends.

Now for the conjunctions and disjunctions.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/conj_disj.html" width="100%" height="500"></iframe>

Does it look like the circled points form a bit of a separate cluster? We thought so, and indeed there's a pattern to those statements. See if you can figure out what it is (answer below).

## More diverse datasets

All of the datasets above were curated to contain statements which are *uncontroversial*, *unambiguous*, and *simple*. They are also not very diverse -- each dataset is formed from a single template.

In contrast, we'll now look at some *uncurated* datasets adapted <a href="https://arxiv.org/abs/2304.13734">from</a> <a href="https://github.com/Algorithmic-Alignment-Lab/CommonClaim/tree/main">other</a> <a href="https://rome.baulab.info/">sources</a>. Mouse over the plots below to get a sampling of the statements in these datasets.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/uncurated_datasets.html" width="100%" height="500"></iframe>

The first thing to note about these visualizations is that we don't see a clear separation into true and false clusters. This is because the datasets are more diverse. Recall that PCA identifies the most salient axes of variation for a dataset; in more diverse datasets, these axes are more likely to encode some truth-independent feature. For instance, the statements in `companies_true_false` are formed using three different templates, and the top 2 principal components mostly encode the difference between these templates. It's quite shocking that `common_claim_true_false`, consisting of statements as diverse as "Rabbits can partially digest memories" or "Dolphins are capable of acts of impressive intelligence" has as much true/false separation as it does.

If we want to see separation into true/false clusters, we can borrow one of the PCA bases identified from our cleaner datasets. For instance, here are our uncurated datasets visualized in the PCA basis extracted from our `cities` dataset.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/uncurated_datasets_cities_basis.html" width="100%" height="500"></iframe>
