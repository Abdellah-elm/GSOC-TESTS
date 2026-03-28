# Test Tasks — GSoC 2026, ML4Sci

**Abdellah Elmlih**

Quantum Machine Learning for Exoplanet Characterization (EXXA × QMLHEP)

---

## Notebooks

| File | Test | What it does |
|------|------|--------------|
| `final test 1.ipynb` | General | Unsupervised clustering of ALMA protoplanetary disk images |
| `final test 2.ipynb` | Image-Based | Autoencoder with accessible 128-d latent space |
| `final test 3.ipynb` | Sequential | Transit light curve classification (classical + quantum) |

---

## How to run

All three notebooks run on Google Colab with a T4 GPU. Each one installs its own dependencies in the first cell.

**Tests 1 & 2** download the ALMA .fits data automatically. If the download fails (Google Drive rate limit), add the shared folder as a shortcut to your Drive:
1. Open https://drive.google.com/drive/folders/1VkS3RHkAjiKjJ6DnZmEKZ_nUv4w6pz7P
2. Right-click → Add shortcut to Drive → My Drive
3. Re-run the notebook

**Test 3** generates its own data with `batman-package`, no external download needed.

---

## Test 1 — General (Clustering)

Loads 38 synthetic ALMA 1250μm continuum images (.fits, layer 0 of a 4×600×600 cube). Trains a convolutional autoencoder without skip connections to extract 128-d feature vectors. Clusters the latent space with K-Means (automatic k via silhouette score) and HDBSCAN on UMAP embeddings. Includes cluster galleries, decoded centroids, and morphological analysis per cluster (brightness, gap contrast, azimuthal asymmetry).

Rotation augmentation during training prevents trivially clustering by viewing angle.

**Inference on new data:**
```python
results = inference_pipeline('./new_fits_folder/')
print(results['cluster_labels'])
```

**Saved files:** `best_feature_extractor.pt`, `cluster_assignments.json`, `latent_vectors.npy`

---

## Test 2 — Image-Based (Autoencoder)

Same data. Convolutional autoencoder with a 128-d bottleneck and no skip connections — the latent vector alone reconstructs the image, which is the point.

Loss: 0.3 × MSE + 0.7 × (1 − MS-SSIM).

**Latent space access:**
```python
z = encode_image('planet0_00226_1250.fits')  # numpy (128,)
img = decode_latent(z)                        # numpy (128, 128)
```

**Inference on new data:**
```python
results = run_inference('./new_fits_folder/')
z = results['latents'][0]
```

**Saved files:** `best_autoencoder.pt`, `image_test_summary.json`

---

## Test 3 — Sequential (Transit Classification)

Generates 6000 synthetic transit light curves using `batman` (Mandel & Agol 2002 model) with 8 physical parameters: Rp/Rs, a/Rs, inclination, eccentricity, argument of periastron, period, and two quadratic limb-darkening coefficients. Noise injected at 5 different levels.

Two classifiers trained:
- **Classical:** 1D residual CNN (195K params)
- **Quantum:** hybrid model — CNN backbone + 6-qubit variational circuit via PennyLane (12K params)

Evaluation includes ROC/AUC, precision-recall, confusion matrices, and a noise robustness sweep across 8 σ levels.

**Inference on new data:**
```python
result = predict_transit(flux_array, model_name='cnn')
print(result['label'], result['probability_transit'])
```

Handles variable-length input (resamples to 300 points).

**Saved files:** `best_transit_cnn.pt`, `best_hybrid_quantum.pt`, `results_summary.json`

---

## Dependencies

Installed automatically in each notebook:

```
torch torchvision pytorch-msssim astropy gdown
pennylane pennylane-lightning batman-package
scikit-learn hdbscan umap-learn
matplotlib seaborn scipy tqdm
```
