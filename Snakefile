configfile: "config/config.yaml"


if not config["pairing"].get("keep_order", False):
    raise ValueError("Final predictor requires RG/RGG first and rAHD second; keep_order must be true.")

rule all:
    input:
        config["outputs"]["prepared_pairs_csv"],
        config["outputs"]["features_csv"]


rule predict_final:
    input:
        config["outputs"]["predictions_csv"],
        config["outputs"]["ranked_predictions_csv"]


rule generate_pairs:
    input:
        fasta=config["input_fasta"]
    output:
        csv=config["outputs"]["prepared_pairs_csv"]
    params:
        mode=config["pairing"]["mode"],
        pairs_file=config["pairing"]["pairs_file"],
        list_a=config["pairing"]["list_a"],
        list_b=config["pairing"]["list_b"],
        allow_self="--allow-self-pairs" if config["pairing"]["allow_self_pairs"] else "",
        keep_order="--keep-order" if config["pairing"].get("keep_order", False) else ""
    shell:
        """
        if [ "{params.mode}" = "explicit_pairs" ]; then \
          python scripts/generate_pairs.py \
            --fasta {input.fasta:q} \
            --pairs-file {params.pairs_file:q} \
            {params.keep_order} \
            --output {output.csv:q}; \
        elif [ "{params.mode}" = "two_lists" ]; then \
          python scripts/generate_pairs.py \
            --fasta {input.fasta:q} \
            --list-a {params.list_a:q} \
            --list-b {params.list_b:q} \
            {params.allow_self} \
            {params.keep_order} \
            --output {output.csv:q}; \
        else \
          echo "Unsupported pairing mode: {params.mode}" >&2; \
          exit 1; \
        fi
        """


rule compute_pair_features:
    input:
        pairs_csv=config["outputs"]["prepared_pairs_csv"],
        fasta=config["input_fasta"],
        global_parameters="resources/global_residues.csv",
        patch_parameters="resources/residues.csv",
        implementation="scripts/feature_engineering.py",
        schema="config/final_feature_schema.json"
    output:
        config["outputs"]["features_csv"]
    shell:
        "python scripts/final_model_adapter.py features --pairs {input.pairs_csv:q} --fasta {input.fasta:q} --output {output:q}"

rule predict_with_hybrid_model:
    input:
        features=config["outputs"]["features_csv"],
        manifest=config["models"]["final_manifest"],
        base="models/final_frozen/base_model_ensemble.joblib",
        prep="models/final_frozen/residual_preprocessor.joblib",
        residual="models/final_frozen/residual_nn.keras"
    output:
        pred=config["outputs"]["predictions_csv"],
        ranked=config["outputs"]["ranked_predictions_csv"]
    shell:
        "python scripts/final_model_adapter.py predict --features {input.features:q} --output {output.pred:q} --ranked {output.ranked:q}"

rule prepare_features:
    input:
        config["outputs"]["features_csv"]

rule predict_pretrained:
    input:
        config["outputs"]["predictions_csv"],
        config["outputs"]["ranked_predictions_csv"]

# Explicit opt-in target; never part of all/prepare_features/predict_final.
if "custom_training" in config:
    custom = config["custom_training"]
    rule train_custom:
        input:
            data=custom["data"],
            script="scripts/train_custom_model.py",
            adapter="scripts/feature_engineering.py",
            global_parameters="resources/global_residues.csv",
            patch_parameters="resources/residues.csv"
        output:
            hashes=custom["output"] + "/SHA256SUMS.json",
            metrics=custom["output"] + "/metrics.csv"
        params:
            directory=custom["output"],
            target=custom["target_column"],
            description=custom["target_description"],
            mode=custom.get("mode", "base"),
            split=custom.get("split_unit", "sequence"),
            epochs=custom.get("residual_epochs", 400)
        shell:
            "python scripts/train_custom_model.py --data {input.data:q} --output {params.directory:q} --target-column {params.target:q} --target-description {params.description:q} --mode {params.mode:q} --split-unit {params.split:q} --residual-epochs {params.epochs}"
