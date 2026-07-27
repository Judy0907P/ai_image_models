# Final Project Presentation Notes

---

## 1. Dataset & Preprocessing

### Mentor (baseline): Pixel Art sprites

- Dataset: Kaggle pixel-art sprites (`ebrahimelgazar/pixel-art`)
- Shape: **16×16×3**



### Mine: CIFAR-10

Dataset: CIFAR-10 (5000 real photos of 10 categories: 

```python
"airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
```

- Preprocess pipeline (`07_final/w1.ipynb`):
  1. Download CIFAR-10 folder structure from Kaggle, load images, stack into arrays
  2. Save in the **same layout** as the mentor format (`sprites.npy`, `sprites_labels.npy`) so the rest of the training stack stays reusable
- Shape: **32×32×3**



## 2. Model Architecture



### Mentor

**FlowMLP**

- Flattens the image, concatenates a sinusoidal **time embedding**, runs an MLP
- Default `img_shape=(16, 16, 3)`, large hidden size (`h=1024`)
- Time conditioning: `cat([z.flatten, time_embed(t)])` at the input

**Simple FlowCNN** (`05_mock_project`)

- Stack of Conv2d + SiLU
- Takes `(z, t)` in `forward` but **ignores** `t`
- Tuned for 16×16 pixel art

**Eval Metric:** FID (Fréchet Inception Distance)via the second last layer of a small classifier model - lower is better.



### Mine

**FlowMLP adaptations**

- Same time-concat MLP idea, but parameterized for **CIFAR size**: `img_shape=(32, 32, 3)`

**FlowCNN with time conditioning** (main design change)

- Problem: mentor CNN never used `t`, so it was not a fair conditional flow model vs the MLP
- Design:
  1. The first 2 convolutional layers extract features from the image only -> (H, W, hidden_channels)
  2. Build `time_embed(t)` as a vector with size t_dim, then use a linear layer to transform it to size hidden_channels, and add dimension for it to become (1, 1, hidden_channels)
  3. **Add** time vector onto image feature
  4. Late 2 convolutional layers map back to normal images with 3 channels-> (H, W, 3)



## 3. Experiments / Sweeps



### Mentor

1. **LR sweep** (`04_experiment_tracking/sweep_lr.yaml`)
  - Grid over `lr ∈ {1e-3, 3e-4, 1e-4}`
  - Goal: learn experiment tracking + FID comparison
2. **Architecture swap** (`05_mock_project/sweep_arch.yaml`)
  - `arch ∈ {mlp, cnn}` with fixed LR



### Mine: combined hyperparameter grid (`07_final/sweep_all.yaml`)

```yaml
arch:        [mlp, cnn]
lr:          [0.001, 0.0003, 0.0001]
batch_size:  [128, 256, 512]
epochs:      [10, 25, 50]
hidden_dim:  [64, 128, 256]
```

- Method: **grid** search in Weights & Biases
- Tracks train loss during learning; logs **FID** at the end of each run
- Ran agents locally and on **Colab** against the same sweep ID so unfinished runs could continue after disconnects
- **NOTE:** Colab GPU limits + a jammed local machine interrupted the sweep. Roughly **80+ MLP** runs finished vs only **20+ CNN** runs

---



## 3.5 Results

1. From the sweep, pick the **best MLP** and **best CNN** by FID
2. Reload those configs and generate sample images (`07_final/w1.ipynb`)


| Arch | Best config (lr, batch, epochs, hidden)              | FID    | Samples                        |
| ---- | ---------------------------------------------------- | ------ | ------------------------------ |
| MLP  | batch_size: 256epochs:10hidden_dim: 128Ir: 0.001  | 27.439 | (images to be added on slides) |
| CNN  | batch_size: 128epochs:10hidden_dim: 256Ir: 0.0003 | 49.288 | (images to be added on slides) |


---



## 4. Learnings & Limitations



### Learnings

- Built an end-to-end pipeline: preprocess a dataset, train a PyTorch model, and generate images
- Practiced core NN ideas (architectures, time conditioning) in a generative setting
- Used sweeps for hyperparameter tuning and FID to compare runs
- Got a concrete feel for how much compute different architectures need (MLP vs CNN, and how quickly a grid explodes)



### Limitations

1. **No class conditioning → blurry “average” images**
When training, we did not provide category info to the model. Also, the style of the images is very different even within each category. Therefore, since the model is trying to learn the style of the entire dataset, it ends up generating somewhat blurry images to minimize the loss.

2. **FID setup is mismatched for this model**
When generating images, we did not provide category either, so the model can generate arbitrary types of images. But the ground-truth features used to calculate FID come from a specific type (the first 2048 images, which are all aircraft). So even if a model generates a perfect picture, as long as it is not an aircraft, the FID will be high — in other words, we would think it is a bad model. So the way we compute FID is not a good indicator of my model’s performance.

3. **Classifier quality was never checked**
We never evaluated the performance of the classifier model, so it is possible that the classifier cannot classify images well. Therefore, using the second-to-last layer to calculate FID may not be accurate.