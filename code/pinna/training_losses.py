"""Training-only objectives: no extra weights, inputs or inference operations."""
import math

import numpy as np
import torch
import torch.nn.functional as F

OBJECTIVES = ('baseline', 'paired_control', 'source_contrastive', 'angular')


def paired_recipe(rng, n_directions, sources, batch):
    """Same direction twice, two different source families; batch audio count fixed."""
    if batch % 2 or batch < 2 or len(sources) < 2:
        raise ValueError('Paired training requires an even batch and at least two source families')
    directions = rng.choice(n_directions, batch//2)
    first = rng.integers(len(sources), size=batch//2)
    second = (first + rng.integers(1, len(sources), size=batch//2)) % len(sources)
    kinds = np.asarray(sources)[np.stack((first, second), 1).ravel()]
    return np.repeat(directions, 2), kinds


def expected_angular_cost(logits, class_vectors, true_vectors):
    """Expected class-to-target angular distance / pi, a differentiable logit cost.

    Geometry is fixed; this is not the angle of a globally averaged prediction.
    It penalizes distant probability mass while retaining the original decoder.
    """
    costs = torch.acos((true_vectors @ class_vectors.T).clamp(-1, 1)) / math.pi
    return (logits.float().softmax(-1) * costs).sum(-1).mean()


def source_contrastive_loss(embedding, labels, vectors, temperature=.1, exclusion_deg=15):
    """SupCon on existing hidden activations, with nearby different directions excluded.

    Same measured direction is a positive; different directions >=15 degrees
    apart are negatives. Intermediate neighbors are excluded from the denominator
    rather than being trained as strongly separated classes. No projection head.
    """
    z = F.normalize(embedding.float(), dim=-1)
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    same = labels[:, None] == labels[None, :]
    positive = same & ~diagonal
    far = (vectors @ vectors.T) <= math.cos(math.radians(exclusion_deg))
    allowed = (same | far) & ~diagonal
    count = positive.sum(1)
    anchors = count > 0
    if not anchors.any():
        return embedding.sum() * 0
    # Every retained anchor has a positive, so its logsumexp is finite.
    scores = z @ z.T / temperature
    selected = scores[anchors]
    logp = selected - torch.logsumexp(selected.masked_fill(~allowed[anchors], -torch.inf), 1, keepdim=True)
    return -(logp.masked_fill(~positive[anchors], 0).sum(1) / count[anchors]).mean()
