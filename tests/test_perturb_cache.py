from src.data.perturbations import apply_field_delta, materialize_perturbed_fields


def test_apply_field_delta_basic():
    base = {"id": "0", "label": "A", "question": "Q", "context": None, "choices": ["x", "y"]}
    delta = {"question": "Q2", "choices": {"1": "y2"}}
    out = apply_field_delta(base, delta)
    assert out["question"] == "Q2"
    assert out["choices"] == ["x", "y2"]


def test_materialize_perturbed_fields_backward_compat():
    clean = [{"id": "0", "label": "A", "question": "Q", "context": None, "choices": ["x", "y"]}]
    # New delta-only format
    rec_new = {"source_idx": 0, "delta": {"question": "Q2"}}
    out_new = materialize_perturbed_fields(clean, [rec_new])
    assert out_new[0]["question"] == "Q2"
    # Old full-field format still works
    rec_old = {"pert_fields": {"id": "0", "label": "A", "question": "Q3", "context": None, "choices": ["x", "y"]}}
    out_old = materialize_perturbed_fields(clean, [rec_old])
    assert out_old[0]["question"] == "Q3"







