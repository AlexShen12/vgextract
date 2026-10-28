# League model experiments — 2026-10-27

These experiments remain separate from production observations. No model quality,
GPU fit or throughput has been measured for the new Midbeast source yet.
Immutable checkpoint/processor revisions are in
`annotation/league/models.local.json`; installed package closures and actual GPU
identity must accompany results.

| Candidate | Local role | Evidence and interpretation limits |
|---|---|---|
| Qwen3.8-27B-FP8 | Production typed crop transcription; separate event baseline | At most four crops/512 output tokens for transcription; 16 frames at 2 FPS for experimental eight-second clips. Thinking disabled. Grammar is not independent verification. |
| InternVL3.5-14B-HF, BF16 | Local event comparison | Same prepared video inputs and visual-only/audio-assisted instructions as Qwen. Report errors and abstentions; no automatic game-state promotion. |
| SAM3.1 | Prompted screen tracking | Five clips, at most three manually prompted objects, native mask references. Mask IoU measures segmentation/tracking, not champion identity, event meaning, minimap position or world coordinates. |
| Qwen3-ASR-1.7B + forced aligner | Commentary pilot | Packaged Silero speech detection, five source-aligned minutes, sequential ASR then alignment. Report WER, name recall, word onset error, latency and memory. Speech can propose windows but cannot establish unseen HUD values or actor identity. |
| Marengo | Retrieval capability review only | Not implemented or invoked; no public offline L40S weights established. |

Marengo uses embeddings to search video fragments by meaning, drawing on visual,
audio and text information. Its documented 3.0 interface embeds video, audio,
images and text; the current documentation also lists 3.5 with document support.
This could help retrieve candidate moments from a large recording library. A
retrieved clip would still require independently validated event/actor extraction;
similarity is not an event label or exact HUD transcription. The documented
platform/API workflow does not establish an offline checkpoint suitable for this
project's L40S allocation. No video was uploaded to that service.
[Official Marengo documentation](https://docs.twelvelabs.io/docs/concepts/models/marengo).

The event protocol requires ten development and ten held-out eight-second clips,
including deaths, recalls, fights, objective/structure changes, quiet negatives,
occlusion and repeated footage. References use source seconds. Report precision,
recall, actor attribution, onset error, clip abstentions, latency and peak GPU
memory separately by split and input modality. Failed/invalid completions are
abstentions, not evidence that an event did not occur. Labels and target answers
never enter prompts. Ground-truth edits reuse compatible raw predictions.

Run GPU models sequentially. The driver closes each node-local vLLM process group
before the next model loads; audio runs in an isolated process and unloads ASR
before the aligner. SAM runs afterward in its own process. Record the actual GPU
name/UUID/capacity; L40 is not L40S. Pinning a compatible architecture and package
version does not establish that kernels, drivers or memory budgets work on the
allocated node; preflight and measured pilots remain required.

Official references:
[Qwen](https://huggingface.co/Qwen/Qwen3.8-27B-FP8),
[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B),
[vLLM video inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/),
[InternVL](https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF),
[SAM](https://github.com/facebookresearch/sam3),
[ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf),
[aligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf).
