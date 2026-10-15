Frozen-model server environment

The two supplied exports are preserved unchanged. frozen_model_linux.yml removes only the machine-specific conda prefix. These exports describe Linux, Python3.10, TensorFlow2.20.0, Keras3.12.0, sklearn1.7.2, NumPy2.2.6. They are not a tested macOS environment.

    conda env create -n idr_frozen -f environments/frozen_model_linux.yml
    conda activate idr_frozen
    python scripts/predict_pairs.py --identity-only --run-name environment_identity

The YAML already includes its pip dependencies; do not blindly install the pip freeze on top. All copied model artifacts match accepted checksums. Runtime prediction identity still must pass in the recreated environment; no local TensorFlow model run was performed.
