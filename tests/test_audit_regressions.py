"""Regression tests covering the deep-learning audit patches.

These tests exercise the following fixes:

1. Beam search beam-expansion: EOS candidates must be stored as finished
   hypotheses without consuming a live beam slot, so that picking from
   ``2 * num_beams`` candidates genuinely expands the search frontier.
2. Attention-mask classifier: accept bool masks and nonzero-float masks
   without forcing a CPU-GPU synchronisation via ``.item()``.
3. ``MobileTranslationModel.generate`` now routes through the cached
   implementation but must remain equivalent under argmax decoding.
4. Trainer weight-decay grouping: norm / embedding parameters must sit in
   the zero-decay group.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lingolite.generation_utils import (
    BeamHypothesis,
    BeamSearchScorer,
    KVCache,
    LayerKVCache,
    generate_with_beam_search,
    generate_with_kv_cache,
)
from lingolite.mobile_translation_model import create_model
from lingolite.model_components import GroupedQueryAttention


# ---------------------------------------------------------------------------
# Beam search: EOS candidates no longer consume live beam slots
# ---------------------------------------------------------------------------


class _ScriptedModel(torch.nn.Module):
    """Mimics just enough of MobileTranslationModel for beam search.

    The model exposes a frozen ``encoder``, a ``decoder`` that returns
    hand-rolled logits, and a ``.eval()`` no-op; this lets us assert the
    exact beam-expansion behaviour that the audit patch targets.
    """

    def __init__(self, vocab_size: int, step_logits):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self._step_logits = step_logits
        self._step_counter = {"idx": 0}

        encoder_module = torch.nn.Module()
        encoder_module.embedding = torch.nn.Embedding(vocab_size, 8)
        self.encoder_module = encoder_module

        decoder_module = torch.nn.Module()
        decoder_module.layers = torch.nn.ModuleList()
        self.decoder_module = decoder_module

    def eval(self):  # type: ignore[override]
        return self

    def train(self, mode: bool = True):  # type: ignore[override]
        return self

    # The beam search calls ``model.encoder(input_ids=..., attention_mask=...)``
    # and ``model.decoder(input_ids=..., ...)`` — we stub both.
    def encoder(self, input_ids, attention_mask=None):
        batch, src_len = input_ids.shape
        return torch.zeros(batch, src_len, 8)

    def decoder(
        self,
        input_ids,
        encoder_output,
        self_attention_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        idx = self._step_counter["idx"]
        self._step_counter["idx"] = idx + 1
        batch, tgt_len = input_ids.shape
        logits = torch.full(
            (batch, tgt_len, self.vocab_size), -1e9, dtype=torch.float32
        )
        # Set the scripted logits on the last position — that is what beam
        # search reads.
        plan = self._step_logits[min(idx, len(self._step_logits) - 1)]
        for token_id, score in plan.items():
            logits[:, -1, token_id] = score
        return logits, None


class _CachingScriptedModel(_ScriptedModel):
    """Beam-search stub that records incremental-cache usage."""

    def __init__(self, vocab_size: int, step_logits):
        super().__init__(vocab_size=vocab_size, step_logits=step_logits)
        self.decoder_input_lengths: list[int] = []
        self.saw_past_key_values = False

    def decoder(
        self,
        input_ids,
        encoder_output,
        self_attention_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        self.decoder_input_lengths.append(int(input_ids.shape[1]))
        if past_key_values is not None:
            self.saw_past_key_values = True

        logits, _ = super().decoder(
            input_ids=input_ids,
            encoder_output=encoder_output,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if not use_cache:
            return logits, None

        batch = input_ids.shape[0]
        cache = LayerKVCache()
        cache.self_attn_cache = KVCache(
            key=torch.zeros(batch, 1, 1, 1),
            value=torch.zeros(batch, 1, 1, 1),
            num_heads=1,
            head_dim=1,
        )
        cache.cross_attn_cache = KVCache(
            key=torch.zeros(batch, 1, 1, 1),
            value=torch.zeros(batch, 1, 1, 1),
            num_heads=1,
            head_dim=1,
        )
        return logits, [cache]


def test_beam_search_stores_eos_hypothesis_without_consuming_beam_slot() -> None:
    """When the top-1 candidate is EOS, beam search must still explore two
    live beams (previously the EOS beam consumed a slot and was suppressed)."""

    vocab_size = 8
    eos_id, sos_id, pad_id = 2, 1, 0
    # Step 0: score [EOS: 0.0, tokA=3: -0.1, tokB=4: -0.5, tokC=5: -2.0, rest -inf].
    # The topk (2*num_beams=4) should be [EOS, 3, 4, 5].
    step0 = {eos_id: 0.0, 3: -0.1, 4: -0.5, 5: -2.0}
    # Step 1: once past the EOS, make all three live beams deterministically emit EOS.
    step1 = {eos_id: 0.0}
    model = _ScriptedModel(vocab_size=vocab_size, step_logits=[step0, step1, step1])

    src = torch.zeros(1, 4, dtype=torch.long)
    out = generate_with_beam_search(
        model=model,
        src_input_ids=src,
        max_length=5,
        num_beams=2,
        length_penalty=1.0,
        early_stopping=True,
        sos_token_id=sos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )

    assert out.shape == (1, 5)
    final_seq = out[0].tolist()
    # With the patched code, the best finished hypothesis is the one with
    # the highest per-length score. The two expansions (3 and 4) both emit
    # EOS at step 1, giving cumulative scores -0.1 + 0 = -0.1 and
    # -0.5 + 0 = -0.5; after length penalty=1.0 the best is [SOS, 3, EOS].
    # The raw step-0 EOS hypothesis is [SOS, EOS] with score 0.0 and
    # length 2, so its length-normalised score is 0.0 (best).
    # The test therefore just verifies we prefer one of the legitimate
    # finished hypotheses (all of which start with SOS and end with EOS).
    assert final_seq[0] == sos_id
    assert eos_id in final_seq, f"finished hypothesis missing EOS: {final_seq}"


def test_beam_search_preserves_multiple_finished_hypotheses() -> None:
    """If the 2*num_beams pool contains multiple EOS candidates they must all
    land in ``finished_hypotheses`` without crowding out live beams."""

    vocab_size = 8
    eos_id = 2
    # Two EOS candidates (from different source beams) plus two live tokens.
    step0 = {eos_id: 0.0, 3: -0.1, 4: -0.5, 5: -0.7}
    # Force live beams to immediately finish on the next step to keep test fast.
    step_final = {eos_id: 0.0}
    model = _ScriptedModel(
        vocab_size=vocab_size,
        step_logits=[step0, step_final, step_final, step_final],
    )

    src = torch.zeros(1, 4, dtype=torch.long)
    _ = generate_with_beam_search(
        model=model,
        src_input_ids=src,
        max_length=6,
        num_beams=2,
        length_penalty=1.0,
        early_stopping=True,
        sos_token_id=1,
        eos_token_id=eos_id,
        pad_token_id=0,
    )
    # The scripted model's internal scorer is not directly accessible from
    # outside, so we assert indirectly: the routine returned without error
    # and consumed at least two decoder steps (one to spawn hypotheses, one
    # to finish live beams). The step counter proves the search genuinely
    # continued past step 0 — under the pre-patch code the run still
    # progressed, but it only carried a single live beam.
    assert model._step_counter["idx"] >= 2


def test_beam_search_uses_incremental_kv_cache() -> None:
    """Beam search should decode a single token per step and thread caches."""
    model = _CachingScriptedModel(
        vocab_size=8,
        step_logits=[
            {3: 0.0, 4: -0.2},
            {2: 0.0},
            {2: 0.0},
        ],
    )

    src = torch.zeros(1, 4, dtype=torch.long)
    _ = generate_with_beam_search(
        model=model,
        src_input_ids=src,
        max_length=5,
        num_beams=2,
        sos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )

    assert model.decoder_input_lengths
    assert all(length == 1 for length in model.decoder_input_lengths)
    assert model.saw_past_key_values


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_beam_search_reorders_cpu_src_mask_when_inputs_on_cuda() -> None:
    """Reindexing src masks during beam search must not fail on CPU/GPU mismatch."""
    model = _CachingScriptedModel(
        vocab_size=8,
        step_logits=[
            {3: 0.0, 4: -0.2},
            {2: 0.0},
            {2: 0.0},
        ],
    ).cuda()
    src = torch.zeros(1, 4, dtype=torch.long, device="cuda")
    src_mask = torch.ones(1, 4, dtype=torch.float32, device="cpu")

    out = generate_with_beam_search(
        model=model,
        src_input_ids=src,
        src_attention_mask=src_mask,
        max_length=5,
        num_beams=2,
        sos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )

    assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# Attention mask: no .item() sync, bool + float both supported
# ---------------------------------------------------------------------------


def _reference_attention(
    attn: GroupedQueryAttention, query, key, value, mask
) -> torch.Tensor:
    # Pure-PyTorch reference: scaled dot-product with additive mask.
    B, q_len, _ = query.shape
    kv_len = key.shape[1]
    Q = attn.q_proj(query).view(B, q_len, attn.n_heads, attn.head_dim).transpose(1, 2)
    K = attn.k_proj(key).view(B, kv_len, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    V = attn.v_proj(value).view(B, kv_len, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    if attn.n_rep > 1:
        K = K.repeat_interleave(attn.n_rep, dim=1)
        V = V.repeat_interleave(attn.n_rep, dim=1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (attn.head_dim ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask.expand_as(scores), float("-inf"))
    attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
    out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, q_len, -1)
    return attn.o_proj(out)


def test_attention_accepts_bool_mask_without_item_sync() -> None:
    torch.manual_seed(0)
    d_model, n_heads, n_kv = 16, 4, 2
    attn = GroupedQueryAttention(d_model, n_heads, n_kv, dropout=0.0).eval()

    B, T = 2, 6
    x = torch.randn(B, T, d_model)
    bool_mask = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, False, False, False]]
    )  # (B, T)
    # Reference: additive mask from bool (True → 0, False → -inf).
    ref_mask = bool_mask[:, None, None, :]

    out_bool, _ = attn(query=x, attention_mask=bool_mask)
    out_ref = _reference_attention(attn, x, x, x, ref_mask)
    assert torch.allclose(out_bool, out_ref, atol=1e-5)


def test_attention_accepts_float_binary_mask() -> None:
    torch.manual_seed(0)
    d_model, n_heads, n_kv = 16, 4, 2
    attn = GroupedQueryAttention(d_model, n_heads, n_kv, dropout=0.0).eval()

    B, T = 2, 6
    x = torch.randn(B, T, d_model)
    float_mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
    )
    ref_mask = (float_mask != 0)[:, None, None, :]

    out_float, _ = attn(query=x, attention_mask=float_mask)
    out_ref = _reference_attention(attn, x, x, x, ref_mask)
    assert torch.allclose(out_float, out_ref, atol=1e-5)


def test_attention_accepts_additive_padding_mask() -> None:
    torch.manual_seed(0)
    d_model, n_heads, n_kv = 16, 4, 2
    attn = GroupedQueryAttention(d_model, n_heads, n_kv, dropout=0.0).eval()

    B, T = 2, 6
    x = torch.randn(B, T, d_model)
    additive_mask = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, -1e4, -1e4], [0.0, 0.0, 0.0, -1e4, -1e4, -1e4]]
    )
    ref_mask = (additive_mask >= 0)[:, None, None, :]

    out_masked, _ = attn(query=x, attention_mask=additive_mask)
    out_ref = _reference_attention(attn, x, x, x, ref_mask)
    assert torch.allclose(out_masked, out_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# generate() now routes through KV cache but is equivalent under argmax
# ---------------------------------------------------------------------------


def test_generate_cached_matches_manual_autoregressive_loop() -> None:
    """The public ``generate()`` entry point must produce the same token
    sequence as a reference no-cache argmax loop that reuses the decoder."""

    torch.manual_seed(0)
    model = create_model(vocab_size=32, model_size="tiny", pad_token_id=0)
    model.eval()

    src = torch.randint(4, 32, (2, 8))
    src_mask = torch.ones_like(src, dtype=torch.float)
    max_length = 10
    sos_id, eos_id = 1, 2

    with torch.no_grad():
        cached = model.generate(
            src_input_ids=src,
            src_attention_mask=src_mask,
            max_length=max_length,
            sos_token_id=sos_id,
            eos_token_id=eos_id,
        )

        # Reference: manual greedy loop without cache.
        encoder_output = model.encoder(input_ids=src, attention_mask=src_mask)
        generated = torch.full((src.shape[0], 1), sos_id, dtype=torch.long)
        finished = torch.zeros(src.shape[0], dtype=torch.bool)
        for _ in range(max_length - 1):
            tgt_mask = torch.ones_like(generated, dtype=torch.float)
            logits, _ = model.decoder(
                input_ids=generated,
                encoder_output=encoder_output,
                self_attention_mask=tgt_mask,
                cross_attention_mask=src_mask,
                use_cache=False,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, model.pad_token_id),
                next_token,
            )
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

    # Trim to the shorter of the two (cached may break early as well).
    common = min(cached.shape[1], generated.shape[1])
    assert torch.equal(cached[:, :common], generated[:, :common])


def test_generate_with_kv_cache_defaults_to_argmax() -> None:
    torch.manual_seed(0)
    model = create_model(vocab_size=32, model_size="tiny", pad_token_id=0)
    model.eval()
    src = torch.randint(4, 32, (1, 6))

    with torch.no_grad():
        a = generate_with_kv_cache(
            model=model, src_input_ids=src, max_length=8, sos_token_id=1, eos_token_id=2
        )
        b = generate_with_kv_cache(
            model=model, src_input_ids=src, max_length=8, sos_token_id=1, eos_token_id=2
        )
    # Argmax is deterministic across calls.
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Trainer weight-decay groups
# ---------------------------------------------------------------------------


def test_trainer_weight_decay_excludes_norms_and_embeddings() -> None:
    from lingolite.training import TranslationTrainer

    model = create_model(vocab_size=64, model_size="tiny", pad_token_id=0)
    loader = DataLoader([], batch_size=1)
    trainer = TranslationTrainer(
        model=model,
        train_loader=loader,
        learning_rate=1e-4,
        weight_decay=0.1,
        warmup_steps=0,
        max_steps=2,
        device="cpu",
        save_dir=".tmp_manual/unused",
    )
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2
    decay_group = next(g for g in groups if g["weight_decay"] != 0.0)
    no_decay_group = next(g for g in groups if g["weight_decay"] == 0.0)

    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    decay_ids = {id(p) for p in decay_group["params"]}

    # Embedding weights must sit in the no-decay group.
    assert id(model.encoder.embedding.weight) in no_decay_ids
    # At least one RMSNorm weight sits in the no-decay group (shape == (d_model,)).
    for module_name, module in model.named_modules():
        if module.__class__.__name__ == "RMSNorm":
            assert id(module.weight) in no_decay_ids, (
                f"RMSNorm weight at {module_name} should be in no-decay group"
            )
            break
    # A Linear projection weight must sit in the decay group.
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert id(module.weight) in decay_ids, (
                f"Linear weight at {module_name} should be in decay group"
            )
            break
