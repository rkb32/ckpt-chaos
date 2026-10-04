import unittest

from ckpt_chaos.invariants import ResumeOutcome, classify


def outcome(**overrides) -> ResumeOutcome:
    base = dict(raised=None, resumed_from_step=10, newest_complete_step=10, weights_diff=0.0, ranks_agree=True)
    base.update(overrides)
    return ResumeOutcome(**base)


class Classify(unittest.TestCase):
    def test_clean_resume_passes(self):
        self.assertEqual(classify(outcome()), "PASS")

    def test_any_exception_is_hard_fail(self):
        self.assertEqual(classify(outcome(raised="FileNotFoundError", resumed_from_step=None, weights_diff=None)), "HARD_FAIL")

    def test_different_weights_are_silent_divergence(self):
        self.assertEqual(classify(outcome(weights_diff=1e-9)), "SILENT_DIVERGENCE")

    def test_ranks_disagreeing_is_divergence_even_if_weights_match(self):
        self.assertEqual(classify(outcome(ranks_agree=False)), "SILENT_DIVERGENCE")

    def test_resuming_from_an_older_checkpoint_is_lost_work(self):
        self.assertEqual(classify(outcome(resumed_from_step=5)), "LOST_WORK")

    def test_divergence_is_never_hidden_behind_lost_work(self):
        self.assertEqual(classify(outcome(resumed_from_step=5, weights_diff=1e-3)), "SILENT_DIVERGENCE")

    def test_resuming_past_the_newest_complete_checkpoint_with_equal_weights_passes(self):
        self.assertEqual(classify(outcome(resumed_from_step=10, newest_complete_step=5)), "PASS")


if __name__ == "__main__":
    unittest.main()
