""" "Utility functions for loss calculations in KGE models."""


def loss_fn_wrapper(model, batch, neg_batch):
    """Wrapper function to compute scores for positive and negative samples.

    :param model: _description_
    :type model: _type_
    :param batch: _description_
    :type batch: _type_
    :param neg_batch: _description_
    :type neg_batch: _type_
    :return: _description_
    :rtype: _type_
    """
    pos_scores = model.score_hrt(batch)
    neg_scores = model.score_hrt(neg_batch)
    return pos_scores, neg_scores
