"""Границы регламента и сохранение проверок при переходе на Draft."""
import asyncio
import pytest
from pydantic import ValidationError
from desk.escalation import Draft, Signals, needs_human, atriage_many
from tests.test_hw2_escalation import NONE, GOOD, SOURCE, draft
from tests.fakes import FakeLLM

@pytest.mark.parametrize('field,amount,expected', [
    ('refund', 5000, False), ('refund', 5001, True),
    ('duplicate', 15000, False), ('duplicate', 15001, True),
    ('refund', 0, False), ('duplicate', None, False),
])
def test_money_boundaries(field, amount, expected):
    assert needs_human(Signals(**{**NONE, field: amount})) is expected

@pytest.mark.parametrize('field', ['fraud', 'threat', 'asks_human', 'tariff_pro', 'merchant_down', 'key_leak'])
def test_independent_reasons(field):
    assert needs_human(Signals(**{**NONE, field: True}))

@pytest.mark.parametrize('change', [
    {'quote': 'выдуманная цитата'}, {'payment_ids': []},
    {'payment_ids': ['P-8815']}, {'severity': 6}, {'amount': -1},
])
def test_draft_checks(change):
    with pytest.raises(ValidationError):
        Draft.model_validate({**GOOD, **change}, context={'source': SOURCE})

def test_batch_keeps_error_and_next_success():
    llm = FakeLLM(['invalid'] * 3 + [draft()])
    out = asyncio.run(atriage_many(llm, [SOURCE, SOURCE], concurrency=1))
    assert isinstance(out[0], Exception)
    assert out[1].needs_human

def test_empty_batch():
    assert asyncio.run(atriage_many(FakeLLM([]), [])) == []


def test_batch_limits_concurrency_and_preserves_input_order():
    import json
    from tests.fakes import text_reply

    class DelayedLLM:
        active = 0
        peak = 0

        async def achat(self, messages, **kwargs):
            text = messages[-1]['content'].split('\n')[1]
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.02 if text == 'первый' else 0.001)
            self.active -= 1
            return text_reply(json.dumps({
                **GOOD, 'quote': text, 'payment_ids': [], 'signals': NONE,
            }, ensure_ascii=False))

    llm = DelayedLLM()
    out = asyncio.run(atriage_many(llm, ['первый', 'второй', 'третий'], concurrency=2))
    assert [t.quote for t in out] == ['первый', 'второй', 'третий']
    assert llm.peak == 2


def test_model_schema_has_no_human_decision():
    from desk.schemas import Ticket
    assert set(Draft.model_fields) == (set(Ticket.model_fields) - {'needs_human'}) | {'signals'}
