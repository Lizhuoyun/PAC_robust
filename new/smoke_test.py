"""
Quick end-to-end smoke test with tiny data to verify the pipeline works.
"""
import os, sys, torch, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_cfg
from model_utils import load_model_and_tokenizer, get_label_token_ids, tokenize_for_classification, extract_class_logits
from data_utils import load_task_data
from perturbations import random_perturbation
from plugin import plugin_loss, compute_margins

def test_pipeline():
    # Use tiny AG News subset
    cfg = get_cfg("agnews", train_size=32, val_size=8, test_size=8, 
                  num_epochs=1, device="cuda:0")
    
    print("Loading model...")
    model, tokenizer, processor = load_model_and_tokenizer(cfg)
    label_token_ids = get_label_token_ids(tokenizer, cfg["label_chars"])
    device = next(model.parameters()).device
    
    print(f"Label token IDs: {dict(zip(cfg['label_chars'], label_token_ids))}")
    
    # Test tokenize
    prompts = ["Classify: Hello world.\nA. World\nB. Sports\nC. Business\nD. Sci/Tech\n\nAnswer:"]
    label_chars = ["A"]
    enc = tokenize_for_classification(tokenizer, prompts, label_chars, max_len=128)
    print(f"Input shape: {enc['input_ids'].shape}")
    print(f"Label positions: {enc['label_positions']}")
    
    # Verify label token is at the right position
    last_tok = enc['input_ids'][0, -1].item()
    print(f"Last token id={last_tok}, expected={label_token_ids[0]}")
    assert last_tok == label_token_ids[0], f"Label token mismatch: {last_tok} != {label_token_ids[0]}"
    
    # Forward pass
    enc_device = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        out = model(input_ids=enc_device["input_ids"], 
                    attention_mask=enc_device["attention_mask"])
    logits = extract_class_logits(out.logits, enc_device["label_positions"], label_token_ids)
    print(f"Class logits shape: {logits.shape}, values: {logits[0].tolist()}")
    print(f"Predicted: {cfg['label_chars'][logits.argmax(dim=-1).item()]}")
    
    # Test training step
    print("\nTesting training step...")
    model.train()
    data = load_task_data(cfg, seed=42)
    batch = data["train"][:4]
    
    from train import train_step_text
    rng = random.Random(42)
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    optimizer.zero_grad()
    
    # Base-clean
    loss, metrics = train_step_text(model, tokenizer, batch, "base_clean",
                                    label_token_ids, cfg, rng)
    loss.backward()
    optimizer.step()
    print(f"Base-clean loss: {loss.item():.4f}")
    
    # Base-aug
    optimizer.zero_grad()
    loss, metrics = train_step_text(model, tokenizer, batch, "base_aug",
                                    label_token_ids, cfg, rng)
    loss.backward()
    optimizer.step()
    print(f"Base-aug loss: {loss.item():.4f}")
    
    # Plugin
    optimizer.zero_grad()
    loss, metrics = train_step_text(model, tokenizer, batch, "plugin",
                                    label_token_ids, cfg, rng, gamma=0.1)
    loss.backward()
    optimizer.step()
    print(f"Plugin loss: {loss.item():.4f}, r_spec: {metrics['r_spec']:.4f}, r_stab: {metrics['r_stab']:.4f}")
    
    print("\n✓ ALL SMOKE TESTS PASSED")
    
    del model, optimizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_pipeline()
