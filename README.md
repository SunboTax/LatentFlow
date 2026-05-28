# LatentFlow

Official PyTorch implementation of **LatentFlow: Discovering Latent Continuous Dynamics across Channels for Multivariate Time-Series Anomaly Detection**.

LatentFlow is a reconstruction-based framework for multivariate time-series anomaly detection. It models channel dependency evolution as a latent continuous dynamic process inspired by the Ornstein-Uhlenbeck process, and uses a self-blind dependency mechanism to reduce diagonal self-dependency bias and emphasize cross-channel anomalies.


## Installation

```bash
pip install -r requirements.txt
```

Evaluation metrics are computed with the external `vus` package.

## Datasets

Datasets files could be downloaded from the [link](https://drive.google.com/file/d/12UXGE4uYv9Y4jTzrgDiOMd8iqZJfneiC/view?usp=sharing) in the paper. The expected directory structure is as follows:

```text
./dataset/
├── Creditcard
│   ├── Creditcard_test_data.npy
│   ├── Creditcard_test_label.npy
│   └── Creditcard_train_data.npy
├── Kitsune
├── MSL
├── PSM
├── SMD
├── SWAT
├── UNSW
└── WADI
```

## Quick Start

Run one experiment:

```bash
# Example command for MSL dataset with Affiliation_F1 metric
python main.py \
      --model_name LatentFlow \
      --dataset MSL \
      --batch_size 256 \
      --lradj type4 \
      --epochs 30 \
      --patience 5 \
      --dropout 0.3  \
      --lr 1e-3 \
      --d_model 128 \
      --window_size 256 \
      --patch_size 32 \
      --patch_stride 16 \
      --k 3 \
      --q 0.02 \
      --metric_mode aff \
      --select_metric Affiliation_F1
```


Run the main reproduction commands:

```bash
bash script/run.sh
```

`script/run.sh` reports both threshold-based Affiliation metrics and threshold-free VUS metrics. For single-mode evaluation, use `script/run_aff.sh` or `script/run_vus.sh`.

## Citation

```bibtex
@inproceedings{latentflow2026,
  title     = {LatentFlow: Discovering Latent Continuous Dynamics across Channels for Multivariate Time-Series Anomaly Detection},
  author    = {Sun, Lijun and Zhang, Shuai and Xue, Xin and Li, Lanhao and Zhou, Haoyi and Li, Jianxin},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  doi       = {https://doi.org/10.1145/3770855.3818034}
}
```

## License

This project is released under the MIT License.
