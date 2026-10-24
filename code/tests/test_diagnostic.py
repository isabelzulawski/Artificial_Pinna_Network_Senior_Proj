import unittest

import torch

from diagnose_phase import template_scores


class DiagnosticTests(unittest.TestCase):
    def test_known_templates_and_no_information(self):
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        x = torch.eye(3)[labels]
        scores = template_scores(x, x, x, labels, 3)
        self.assertEqual(scores["accuracy_pct"], 100)
        self.assertEqual(scores["shifted_accuracy_pct"], 100)
        self.assertEqual(scores["confusion"], [[2,0,0], [0,2,0], [0,0,2]])
        scores = template_scores(x*0, x*0, x*0, labels, 3)
        self.assertAlmostEqual(scores["accuracy_pct"], 100/3, places=4)


if __name__ == "__main__":
    unittest.main()
