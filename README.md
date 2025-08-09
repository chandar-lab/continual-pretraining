# Continual Pre-Training with Replay + Gradient Alignment (GPT-NeoX)

This repository provides a **GPT-NeoX/Megatron-DeepSpeed** pipeline to perform **continual pre-training (CPT)** with:
- **Experience Replay** using a disk-backed buffer (async prefetch + RAM cache) mixed into each batch.
- **Gradient Alignment** via lightweight **Reptile/MER** meta-updates applied at a configurable cadence.

> If you use this repository, please cite the accompanying paper:
> **“Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models” (2025).**
> See the **📚 Citation** section below.

---

## Highlights

- Drop-in **mixed-batch replay** that plugs into NeoX’s dataloader loop.
- **Disk-resident** buffer with streaming writes, and prefetch to hide I/O latency.
- **MER/Reptile** hook that interpolates weights every *k* steps with negligible compute/memory overhead.
- Metrics scripts for **Forgetting Score**, **Retained Loss**, and **Learned Loss**, plus **lm-eval** integration.



@inproceedings{abbes2025revisiting,
  title        = {Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models},
  author       = {Istabrak Abbes and Gopeshh Subbaraj and Matthew Riemer and Nizar Islah and Benjamin Therien and Tsuguchika Tabaru and Hiroaki Kingetsu and Sarath Chandar and Irina Rish},
  booktitle    = {Conference on Lifelong Learning Agents (CoLLAs)},
  year         = {2025},
  archivePrefix= {arXiv},
  eprint       = {2508.01908}
}