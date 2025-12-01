# RewACT Pipeline - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Modal
```bash
pip install modal
modal setup
```

### 2. Configure Secrets
```bash
./scripts/setup_modal.sh
```

This will prompt you for:
- HuggingFace token (`hf_...`)
- WandB API key
- Discord webhook URL (optional)

### 3. Run the Pipeline
```bash
modal run scripts/modal_pipeline.py
```

That's it! The entire training loop will run in the cloud.

## 📱 Monitor Progress

You'll receive Discord notifications for:
- 🚀 Pipeline start
- 🧮 Each step start (with links to Modal & WandB)
- ✅ Each step completion (with HF Hub links)
- 🎉 Pipeline completion

## 📊 What Happens

```
Step 1: Train Value Function (A100, ~4-6 hours)
   ↓
Step 2: Compute Advantages (T4, ~30 min per dataset, parallel)
   ↓
Step 3: Train ACTvantage Policy (A100, ~8-12 hours)
   ↓
Done! All outputs on HuggingFace Hub
```

## 🎯 Common Use Cases

### Use Existing Value Function
```bash
modal run scripts/modal_pipeline.py \
  --skip-value-training \
  --value-function-repo "username/my-existing-model"
```

### Custom Datasets
```bash
modal run scripts/modal_pipeline.py \
  --datasets "user/dataset1,user/dataset2,user/dataset3"
```

### Push Checkpoints Every 10k Steps
```bash
modal run scripts/modal_pipeline.py \
  --checkpoint-push-freq 10000
```

### Quick Test Run
```bash
modal run scripts/modal_pipeline.py \
  --value-steps 1000 \
  --policy-steps 1000
```

## 📦 Outputs

All outputs automatically pushed to HuggingFace Hub:
- **Value Function**: `{value-function-repo}`
- **Advantages**: `{dataset}-advantages` (one per dataset)
- **Policy**: `{actvantage-repo}`
- **Checkpoints** (if enabled): `{repo}-checkpoint-{step}`

## 🔧 Troubleshooting

**Pipeline fails midway?**
```bash
# Resume using skip flags
modal run scripts/modal_pipeline.py --skip-value-training --skip-advantage-computation
```

**Discord not working?**
```bash
# Test your webhook
python scripts/test_discord.py
```

**Check secrets:**
```bash
modal secret list
```

## 📚 Documentation

- **Full guide**: See `MODAL_PIPELINE.md`
- **Implementation details**: See `PIPELINE_SUMMARY.md`
- **Examples**: See `scripts/pipeline_config_example.py`

## 💡 Key Benefits

✅ **No babysitting** - Walk away, get notified when done  
✅ **Cloud GPUs** - A100 for training, T4 for inference  
✅ **Cost efficient** - Pay only for compute time, no idle costs  
✅ **Robust** - Resume from any step if something fails  
✅ **Automatic saving** - All outputs to HuggingFace Hub  
✅ **Parallel processing** - Advantages computed in parallel  

## 🎓 Learn More

**View your runs**: https://modal.com/apps  
**Modal docs**: https://modal.com/docs  
**Get help**: Check troubleshooting in `MODAL_PIPELINE.md`

---

**Next step**: Run `modal run scripts/modal_pipeline.py` and watch the magic happen! ✨

