# Published artifacts

Hosted companion pages for operating and demoing this stack. Both are generated from
the docs in this repo and carry the same content in a form that is easier to read on a
phone or a second monitor while the terminal is busy.

| Page | What it is | Source in repo |
|---|---|---|
| [vLLM Box Commands](https://claude.ai/code/artifact/a2c27224-3c83-4448-810a-04c13bc86cc7) | Every command, tagged by whether it runs on the GPU box or your laptop, with copy buttons | [`docs/commands.md`](commands.md) |
| [vLLM Run of Show](https://claude.ai/code/artifact/0726d40c-4e4c-4a3d-b304-04b7b2c85bf4) | Nine-beat recording script: what to do on screen and what to say while it happens | this file |

**These links are private to the repo owner's account** unless explicitly shared. The
markdown sources in `docs/` are the canonical, always-readable copies. Treat the hosted
pages as a convenience layer, not the source of truth.

## Why the command page is split by location

The single most expensive mistake when working a rented GPU box is running a command in
the wrong terminal: the laptop and the box both present a shell prompt, and `apt-get`
failing silently on macOS looks a lot like a broken box. Every command block on that page
is tagged **BOX** or **MAC** for that reason. When in doubt, `hostname`.

## Why the run of show closes on limitations

The last beat of the recording names what the project does not prove: one model, one
GPU, ninety seconds of synthetic load, a single shared API key, no per-user quotas, no
rate limiting. That is deliberate. Stating the limits is more credible than claiming
production-readiness the demo does not demonstrate, and it pre-empts the first question
a reviewer would ask anyway.
