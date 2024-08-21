# Listwise Generative Retrieval Models via a Sequential Learning Process

## Abstract
Recently, a novel generative retrieval (GR) paradigm has been proposed, where a single sequence-to-sequence model is learned to directly generate a list of relevant document identifiers (docids) given a query. Existing GR models commonly employ maximum likelihood estimation (MLE) for optimization: this involves maximizing the likelihood of a single relevant docid given an input query,  with the assumption that the likelihood for each docid is independent of the other docids in the list. We refer to these models as the pointwise approach in this paper. While the pointwise approach has been shown to be effective in the context of GR, it is considered sub-optimal due to its disregard for the fundamental principle that ranking involves making predictions about lists. In this paper, we address this limitation by introducing an alternative listwise approach, which empowers the GR model to optimize the relevance at the docid list level. Specifically, we view the generation of a ranked docid list as a sequence learning process: at each step we learn a subset of parameters that maximizes the corresponding generation likelihood of the $i$-th docid given the (preceding) top $i-1$ docids. To formalize the sequence learning process, we design a positional conditional probability for GR. To alleviate the potential impact of beam search on the generation quality during inference, we perform relevance calibration on the generation likelihood of model-generated docids according to relevance grades. We conduct extensive experiments on representative binary and multi-graded relevance datasets. Our empirical results demonstrate that our method outperforms state-of-the-art GR baselines in terms of retrieval performance.


![Optimization objectives. Assume that the following are given: a query $q$ and two ground-truth docids, $docid_1$ and $docid_2$, where $docid_1$ is more relevant than $docid_2$ to $q$. Top: Most existing GR work relies on maximum likelihood estimation, by maximizing the likelihood of the target docid for each query-docid pair. All relevant docids $docid_1$ and $docid_2$ are treated equally, sharing similar likelihood values. Bottom: A listwise objective (yellow rectangle) is designed for GR, directly modeling the ranked docid lists and incorporating positional information between $docid_1$ and $docid_2$ ($docid_1$ with darker green has a larger positional weight), resulting in a positional weighted likelihood.](/workspaces/ListGR/resources/contrast.png)

## Approach overview

In this paper, we propose a listwise GR approach (ListGR for short), in which docid lists instead of individual docids are used as instances in learning. ListGR includes a two-stage optimization process, i.e., training with position-aware ListMLE and re-training with relevance calibration. 

![Overview of the two-stage listwise learning methods, which consists of a training stage using listwise loss and a re-training stage with relevance calibration based on the trained model.](resources/two-stage.pdf)

## Resources

[Paper](resources/ListGR-with-DOI.pdf)
