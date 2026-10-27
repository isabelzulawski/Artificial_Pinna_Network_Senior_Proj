import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinna import config, model, train
from pinna.training_losses import expected_angular_cost, paired_recipe, source_contrastive_loss


class TrainingLossTests(unittest.TestCase):
    def test_paired_sources_and_directions(self):
        a = paired_recipe(np.random.default_rng(3), 20, ('white', 'pink', 'band'), 16)
        b = paired_recipe(np.random.default_rng(3), 20, ('white', 'pink', 'band'), 16)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[0][::2], a[0][1::2])
        self.assertTrue(np.all(a[1][::2] != a[1][1::2]))
        with self.assertRaises(ValueError):
            paired_recipe(np.random.default_rng(3), 20, ('white',), 16)

    def test_angular_cost_geometry_and_gradients(self):
        v = torch.tensor([[1., 0, 0], [0, 1., 0], [-1., 0, 0]])
        values = []
        for i in range(3):
            logits = torch.full((1, 3), -20.)
            logits[0, i] = 20.
            values.append(float(expected_angular_cost(logits, v, v[:1])))
        np.testing.assert_allclose(values, [0, .5, 1], atol=1e-6)
        logits = torch.zeros(1, 3, requires_grad=True)
        expected_angular_cost(logits, v, v[:1]).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0, 0]), float(logits.grad[0, 2]))

    def test_contrastive_alignment_and_degenerate_batch(self):
        labels = torch.tensor([0, 0, 1, 1])
        v = torch.eye(3)[labels]
        good = torch.tensor([[1., 0], [1., 0], [0, 1.], [0, 1.]], requires_grad=True)
        bad = torch.tensor([[1., 0], [0, 1.], [1., 0], [0, 1.]])
        loss = source_contrastive_loss(good, labels, v)
        self.assertLess(float(loss.detach()), float(source_contrastive_loss(bad, labels, v)))
        loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())
        # A different direction only five degrees away must not become a hard negative.
        angle = np.radians(5)
        nearby = torch.tensor([[1., 0, 0], [1., 0, 0], [np.cos(angle), np.sin(angle), 0],
                               [np.cos(angle), np.sin(angle), 0]], dtype=torch.float32)
        self.assertAlmostEqual(float(source_contrastive_loss(bad, labels, nearby)), 0, places=6)
        unique = torch.randn(3, 4, requires_grad=True)
        zero = source_contrastive_loss(unique, torch.arange(3), torch.eye(3))
        zero.backward()
        self.assertTrue(torch.isfinite(unique.grad).all())

    def test_all_objectives_same_model_and_checkpoint(self):
        ir = np.zeros((3, 80), np.float32)
        ir[:, 10] = 1
        ir[1, 22], ir[2, 37] = .4, -.3
        v = torch.eye(3, device=config.DEVICE)
        counts = []
        for objective in ('baseline', 'paired_control', 'source_contrastive', 'angular'):
            cfg = train.TrainConfig(training_objective=objective, select_by='angular',
                                    steps=2, batch=4, eval_every=1, sources=('white', 'pink'), preroll=128)
            fitted, history, info = train.fit(ir, v, np.arange(3), cfg, log=lambda _: None)
            counts.append(info['n_params'])
            self.assertEqual(info['selection_metric'], 'val_err_mean')
            self.assertTrue(np.isfinite(info['best_val_err_mean']))
            x = torch.randn(2, 191, device=config.DEVICE)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/'model.pt'
                model.save(path, fitted, info=info)
                restored, _ = model.load(path, config.DEVICE)
                torch.testing.assert_close(fitted(x), restored(x))
        self.assertEqual(len(set(counts)), 1)
        self.assertEqual(sum(p.numel() for p in model.DirectionMLP(191, 632).parameters()), 277750)


if __name__ == '__main__':
    unittest.main()
