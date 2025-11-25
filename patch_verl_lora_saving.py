#!/usr/bin/env python3
"""
Patch VERL's FSDPEngine to save LoRA adapters in checkpoints.

This adds the missing LoRA adapter saving logic to the hybrid_engine code path.
"""

import os
import sys

def patch_fsdp_engine():
    """Add LoRA adapter saving to FSDPEngine.save_checkpoint()"""
    
    # Find the file to patch
    import verl
    verl_path = os.path.dirname(verl.__file__)
    target_file = os.path.join(verl_path, "workers/engine/fsdp/transformer_impl.py")
    
    if not os.path.exists(target_file):
        print(f"❌ Error: Could not find {target_file}")
        return False
    
    # Read the file
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "Save LoRA adapter if using LoRA" in content:
        print("✅ Already patched! LoRA saving is enabled.")
        return True
    
    # Find the save_checkpoint method and add LoRA saving
    original_pattern = """        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint("""
    
    patched_code = """        torch.distributed.barrier()
        
        # Save LoRA adapter if using LoRA
        lora_rank = getattr(self.model_config, 'lora_rank', 0)
        lora_adapter_path = getattr(self.model_config, 'lora_adapter_path', None)
        is_lora = lora_adapter_path is not None or lora_rank > 0
        
        if is_lora:
            import json
            from dataclasses import asdict
            from verl.utils.logger import log_with_rank
            from safetensors.torch import save_file
            from verl.utils.fsdp_utils import layered_summon_lora_params, fsdp_version
            
            # Get the model, unwrapping FSDP if needed
            peft_model = self.module
            if fsdp_version(self.module) == 1:
                unwrapped = self.module._fsdp_wrapped_module
            else:
                unwrapped = self.module
            
            # Check if it actually has LoRA/PEFT config
            if hasattr(unwrapped, "peft_config"):
                lora_save_path = os.path.join(local_path, "lora_adapter")
                peft_config = {}
                
                if torch.distributed.get_rank() == 0:
                    os.makedirs(lora_save_path, exist_ok=True)
                    peft_config = asdict(unwrapped.peft_config.get("default", {}))
                    peft_config["task_type"] = peft_config["task_type"].value
                    peft_config["peft_type"] = peft_config["peft_type"].value
                    peft_config["target_modules"] = list(peft_config["target_modules"])
                
                try:
                    lora_params = layered_summon_lora_params(self.module)
                    if torch.distributed.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
                    log_with_rank(
                        f"Saved LoRA adapter to: {lora_save_path}",
                        rank=torch.distributed.get_rank(),
                        logger=logger,
                        log_only_rank_0=True
                    )
                except Exception as e:
                    log_with_rank(
                        f"Save LoRA Adapter Error ({e})",
                        rank=torch.distributed.get_rank(),
                        logger=logger,
                        log_only_rank_0=True
                    )
                
                torch.distributed.barrier()
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint("""
    
    if original_pattern not in content:
        print("❌ Error: Could not find the expected code pattern to patch.")
        print("   VERL version might have changed. Manual patching required.")
        return False
    
    # Apply the patch
    patched_content = content.replace(original_pattern, patched_code)
    
    # Create backup
    backup_file = target_file + ".backup"
    if not os.path.exists(backup_file):
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"✅ Created backup: {backup_file}")
    
    # Write patched file
    with open(target_file, 'w') as f:
        f.write(patched_content)
    
    print(f"✅ Successfully patched: {target_file}")
    print("✅ LoRA adapters will now be saved in checkpoint folders!")
    return True

def verify_patch():
    """Verify the patch was applied correctly"""
    import verl
    verl_path = os.path.dirname(verl.__file__)
    target_file = os.path.join(verl_path, "workers/engine/fsdp/transformer_impl.py")
    
    with open(target_file, 'r') as f:
        content = f.read()
    
    if "Save LoRA adapter if using LoRA" in content:
        print("\n✅ Verification PASSED: Patch is applied correctly")
        return True
    else:
        print("\n❌ Verification FAILED: Patch not found in file")
        return False

def restore_backup():
    """Restore from backup if something went wrong"""
    import verl
    verl_path = os.path.dirname(verl.__file__)
    target_file = os.path.join(verl_path, "workers/engine/fsdp/transformer_impl.py")
    backup_file = target_file + ".backup"
    
    if not os.path.exists(backup_file):
        print("❌ No backup file found")
        return False
    
    with open(backup_file, 'r') as f:
        content = f.read()
    
    with open(target_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Restored from backup: {backup_file}")
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("VERL LoRA Checkpoint Saving Patcher")
    print("=" * 70)
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        print("Restoring from backup...")
        restore_backup()
    else:
        success = patch_fsdp_engine()
        if success:
            verify_patch()
            print()
            print("=" * 70)
            print("Next steps:")
            print("1. Your VERL installation is now patched")
            print("2. Run your training as normal with hybrid_engine: true")
            print("3. Checkpoints will include both:")
            print("   - model_world_size_X_rank_Y.pt (FSDP shards)")
            print("   - lora_adapter/ (LoRA weights)")
            print()
            print("To restore original VERL:")
            print(f"  python {__file__} --restore")
            print("=" * 70)
        else:
            print()
            print("=" * 70)
            print("Patch failed. You may need to:")
            print("1. Check VERL version compatibility")
            print("2. Manually apply the patch")
            print("3. Report this as a bug to VERL maintainers")
            print("=" * 70)
            sys.exit(1)

