# The Geometry of Truth: Dataexplorer

This page contains a plethora of interactive charts for exploring the data from the paper *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets* by Samuel Marks and Max Tegmark.

The charts on this page are all produced simply by performing PCA to visualize high-dimensional LLaMA-13B representations of our datasets.

## Basic datasets

Let's start off with our basic datasets, containing simple statements like "The city of Beijing is in China" (true) or "Fifty-eight is larger than sixty-one" (false). You can see which statements the points shown correspond to by mousing over.

<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/basic_datasets.html" width="1200" height="500"></iframe>

We're not sure why the `smaller_than` dataset doesn't look as separated as the rest. But things look better when you go to 3D (below, right).
<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/cities_3D.html" width="500", height="500"></iframe>
<iframe src="https://saprmarks.github.io/geometry-of-truth/dataexplorer/plots/cities_3D.html" width="500", height="500"></iframe>

Even with these simple plots, there's already lots to explore! For instance, for `larger_than`, can you figure out how the statements vary as you run move along the point clouds (up and to the right)? See below for the answer.