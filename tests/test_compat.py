import unittest

from safety_guard.compat import torch_load_compat


class CompatTests(unittest.TestCase):
    def test_torch_load_compat_retries_without_weights_only_for_old_torch(self):
        calls = []

        def fake_load(path, **kwargs):
            calls.append(kwargs)
            if "weights_only" in kwargs:
                raise TypeError("'weights_only' is an invalid keyword argument")
            return "loaded"

        self.assertEqual(torch_load_compat("x.pt", torch_load=fake_load), "loaded")
        self.assertEqual(calls, [{"weights_only": False}, {}])


if __name__ == "__main__":
    unittest.main()
