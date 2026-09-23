# LDSR-S2 configuration

`ldsrs2.yaml` is copied unchanged from the installed `opensr-model==1.1.1` package's `configs/config_10m.yaml`. The architecture and pretrained checkpoint are unchanged. The application now defaults to the upstream **100** sampling steps. Per-job options of 20, 50 and 100 are applied under the model lock and stored with each result.

Checkpoint: `opensr-ldsrs2_v1_0_0.ckpt`

SHA-256: `e2621e3912eb7c14867c3d20c9029607ba941be8e166dc09621860fcac27dc3a`

The application reads this tracked config locally; startup no longer fetches a moving GitHub configuration. Downloading the named checkpoint is an explicit setup command with checksum verification.
