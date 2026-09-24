"""Candidates sent to the routing service must be clean: unknown prices are not free,
duplicate ids are sent once, and missing catalog fields are not silently zeroed."""

import unittest
from unittest.mock import patch

from model_router import Limits, ModelInfo, NoModelFitsError, Router
from model_router.jev import _describe

RAW = {
    "id": "acme/thing",
    "context_length": 128_000,
    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
    "top_provider": {"max_completion_tokens": None},
    "description": "",
    "created": 0,
    "expiration_date": None,
}

CATALOG = {
    "acme/free": ModelInfo("acme/free", 128_000, None, 0.0, 0.0, "truly free"),
    "acme/mystery": ModelInfo("acme/mystery", 128_000, None, 0.0, 0.0, "", price_known=False),
    "acme/noctx": ModelInfo("acme/noctx", 0, None, 1e-6, 1e-6, "", context_length_known=False),
    "acme/zeroctx": ModelInfo("acme/zeroctx", 0, None, 1e-6, 1e-6, ""),
    "acme/paid": ModelInfo("acme/paid", 128_000, None, 1e-6, 1e-6, "", created=10),
    "acme/new": ModelInfo("acme/new", 128_000, None, 1e-6, 1e-6, "", created=20),
}


def make_router(models=None, providers=None, limits=None):
    with patch("model_router.router.fetch_catalog", return_value=CATALOG):
        return Router(
            openrouter_api_key="k",
            models=models,
            providers=providers,
            limits=limits or Limits(),
        )


class UnknownPriceTest(unittest.TestCase):
    def test_missing_price_is_unknown_not_free(self):
        m = ModelInfo.from_openrouter({k: v for k, v in RAW.items() if k != "pricing"})
        self.assertFalse(m.price_known)
        self.assertIsNone(m.cost(100, 100))

    def test_negative_variable_price_is_unknown_not_free(self):
        m = ModelInfo.from_openrouter({**RAW, "pricing": {"prompt": "-1", "completion": "-1"}})
        self.assertFalse(m.price_known)
        self.assertIsNone(m.cost(100, 100))

    def test_zero_price_is_truly_free(self):
        m = ModelInfo.from_openrouter({**RAW, "pricing": {"prompt": "0", "completion": "0.0"}})
        self.assertTrue(m.price_known)
        self.assertEqual(m.cost(100, 100), 0.0)

    def test_cost_cap_rejects_unknown_price_with_reason(self):
        r = make_router(models=["acme/mystery"], limits=Limits(max_cost_usd=1.0))
        self.assertEqual(r.fitting("hi"), [])
        with self.assertRaises(NoModelFitsError) as ctx:
            r.route("hi")
        self.assertIn("price unknown", str(ctx.exception))

    def test_cost_cap_allows_truly_free(self):
        r = make_router(models=["acme/free"], limits=Limits(max_cost_usd=0.0))
        self.assertEqual([m.id for m in r.fitting("hi")], ["acme/free"])

    def test_unknown_price_passes_when_no_cost_cap(self):
        r = make_router(models=["acme/mystery"])
        self.assertEqual([m.id for m in r.fitting("hi")], ["acme/mystery"])

    def test_describe_shows_unknown_not_zero(self):
        line = _describe(CATALOG["acme/mystery"], 100, Limits(output_tokens=100), 0)
        self.assertIn("unknown", line)
        self.assertNotIn("$0.00", line)
        self.assertNotIn("est. $0", line)

    def test_describe_free_still_shows_zero_price(self):
        line = _describe(CATALOG["acme/free"], 100, Limits(output_tokens=100), 0)
        self.assertIn("$0.00/M in", line)
        self.assertIn("est. $0.000000", line)


class DuplicateCandidateTest(unittest.TestCase):
    def test_same_id_listed_twice_is_kept_once(self):
        r = make_router(models=["acme/paid", "acme/paid"])
        self.assertEqual([m.id for m in r.models], ["acme/paid"])

    def test_repeated_provider_is_kept_once(self):
        r = make_router(providers=["acme", "acme"])
        ids = [m.id for m in r.models]
        self.assertEqual(len(ids), len(set(ids)))

    def test_first_occurrence_wins(self):
        r = make_router(models=["acme/paid", "acme/new", "acme/paid"])
        self.assertEqual([m.id for m in r.models], ["acme/paid", "acme/new"])

    def test_models_and_providers_together_send_each_id_once(self):
        r = make_router(models=["acme/new"], providers=["acme"])
        self.assertEqual([m.id for m in r.models], ["acme/new"])

    @patch("model_router.jev.request_json", return_value={"code": 0, "data": {"decision": "acme/new"}})
    def test_jev_receives_each_candidate_once(self, req):
        with patch("model_router.router.fetch_catalog", return_value=CATALOG):
            r = Router(jev_api_key="k", models=["acme/new", "acme/paid", "acme/new"])
        r.route("hi")
        ids = [c["id"] for c in req.call_args.args[2]["candidates"]]
        self.assertEqual(ids, ["acme/new", "acme/paid"])


class MissingContextLengthTest(unittest.TestCase):
    def test_missing_context_length_is_distinguishable_from_zero(self):
        missing = ModelInfo.from_openrouter({k: v for k, v in RAW.items() if k != "context_length"})
        explicit = ModelInfo.from_openrouter({**RAW, "context_length": 0})
        self.assertEqual((missing.context_length, missing.context_length_known), (0, False))
        self.assertEqual((explicit.context_length, explicit.context_length_known), (0, True))

    def test_missing_context_length_rejects_with_catalog_reason(self):
        r = make_router(models=["acme/noctx"])
        with self.assertRaises(NoModelFitsError) as ctx:
            r.route("hi")
        self.assertIn("catalog provides none", str(ctx.exception))

    def test_explicit_zero_context_rejects_with_context_reason(self):
        r = make_router(models=["acme/zeroctx"])
        with self.assertRaises(NoModelFitsError) as ctx:
            r.route("hi")
        self.assertIn("context 0 <", str(ctx.exception))

    def test_describe_shows_unknown_context(self):
        line = _describe(CATALOG["acme/noctx"], 100, Limits(output_tokens=100), 0)
        self.assertIn("context unknown", line)


if __name__ == "__main__":
    unittest.main()
