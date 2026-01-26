import os

from src.data.prompts import PromptRenderer, PromptTemplate
from src.data.perturbations import build_perturbed_fields_cache, load_perturbed_fields_cache


def test_template_immutability(tmp_path):
    template = PromptTemplate(answer_marker="Answer: ")
    renderer = PromptRenderer(template, labels=["A", "B"])
    fields = {"id": "1", "label": "A", "question": "What is 2+2?", "choices": ["4", "5"], "context": None}
    prompt = renderer.render(fields)
    pert_fields = dict(fields)
    pert_fields["question"] = "What is two plus two?"
    pert_fields["choices"] = ["four", "five"]
    pert_prompt = renderer.render(pert_fields)
    assert "Answer: " in prompt
    assert "Answer: " in pert_prompt
    # Perturbations may change question/choices, but must never change the marker string or its placement.
    assert template.answer_marker == "Answer: "
    assert prompt.endswith("Answer: ")
    assert pert_prompt.endswith("Answer: ")
    assert prompt.count("Answer: ") == 1
    assert pert_prompt.count("Answer: ") == 1


def test_cache_reuse_and_determinism(tmp_path):
    fields = [
        {"id": "1", "label": "A", "question": "Test one", "choices": ["x", "y"], "context": None},
        {"id": "2", "label": "B", "question": "Test two", "choices": ["m", "n"], "context": None},
    ]
    cfg = {"budget": "small", "mix": {"typo": 1.0}}
    path1 = build_perturbed_fields_cache(fields, "arc", "train", cfg, seed=123, n_variants=1, cache_root=str(tmp_path))
    path2 = build_perturbed_fields_cache(fields, "arc", "train", cfg, seed=123, n_variants=1, cache_root=str(tmp_path))
    assert path1 == path2
    rec1 = load_perturbed_fields_cache(path1)
    rec2 = load_perturbed_fields_cache(path2)
    assert rec1 == rec2
