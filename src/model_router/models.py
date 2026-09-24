from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_length: int
    max_output_tokens: int | None  # None = provider sets no separate cap
    prompt_price: float  # USD per input token; 0.0 with price_known=False means "unknown", not free
    completion_price: float  # USD per output token; same caveat
    description: str = ""
    created: int = 0  # unix time the model was listed; used to order a provider's models newest first
    expires: str | None = None  # retirement date if OpenRouter has scheduled one
    price_known: bool = True  # False = catalog price missing or variable (negative); never treat as free
    context_length_known: bool = True  # False = catalog entry carried no context_length at all

    def cost(self, in_tokens, out_tokens):
        """Estimated USD for the task, or None when the catalog price is unknown."""
        if not self.price_known:
            return None
        return in_tokens * self.prompt_price + out_tokens * self.completion_price

    @staticmethod
    def _price(raw_value):
        """(value, known): missing or negative ("variable") prices are unknown, not free."""
        if raw_value is None:
            return 0.0, False
        value = float(raw_value)
        return (value, True) if value >= 0 else (0.0, False)

    @classmethod
    def from_openrouter(cls, raw):
        pricing = raw.get("pricing") or {}
        top = raw.get("top_provider") or {}
        prompt_price, prompt_known = cls._price(pricing.get("prompt"))
        completion_price, completion_known = cls._price(pricing.get("completion"))
        context = raw.get("context_length")
        return cls(
            id=raw["id"],
            context_length=int(context) if context is not None else 0,
            max_output_tokens=top.get("max_completion_tokens"),
            prompt_price=prompt_price,
            completion_price=completion_price,
            description=(raw.get("description") or "").replace("\n", " "),
            created=int(raw.get("created") or 0),
            expires=raw.get("expiration_date"),
            price_known=prompt_known and completion_known,
            context_length_known=context is not None,
        )


@dataclass(frozen=True)
class Limits:
    output_tokens: int = 1024
    max_cost_usd: float | None = None
