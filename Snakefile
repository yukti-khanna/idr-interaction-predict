configfile: "config/config.yaml"


rule all:
    input:
        config["outputs"]["prepared_pairs_csv"],
        config["outputs"]["features_csv"]


rule predict_pretrained:
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
            --fasta {input.fasta} \
            --pairs-file {params.pairs_file} \
            {params.keep_order} \
            --output {output.csv}; \
        elif [ "{params.mode}" = "two_lists" ]; then \
          python scripts/generate_pairs.py \
            --fasta {input.fasta} \
            --list-a {params.list_a} \
            --list-b {params.list_b} \
            {params.allow_self} \
            {params.keep_order} \
            --output {output.csv}; \
        else \
          echo "Unsupported pairing mode: {params.mode}" >&2; \
          exit 1; \
        fi
        """


rule compute_pair_features:
    input:
        pairs_csv=config["outputs"]["prepared_pairs_csv"],
        fasta=config["input_fasta"],
        residues=config["resources"]["residues_file"]
    output:
        config["outputs"]["features_csv"]
    shell:
        """
        python scripts/compute_pair_features.py \
          --pairs-csv {input.pairs_csv} \
          --fasta {input.fasta} \
          --residues-file {input.residues} \
          --output-csv {output}
        """


rule predict_with_hybrid_model:
    input:
        features=config["outputs"]["features_csv"],
        bundle=config["models"]["pretrained_bundle"]
    output:
        pred=config["outputs"]["predictions_csv"],
        ranked=config["outputs"]["ranked_predictions_csv"]
    shell:
        """
        python scripts/predict_with_hybrid_model.py \
          --features-csv {input.features} \
          --bundle {input.bundle} \
          --output-csv {output.pred} \
          --ranked-output-csv {output.ranked}
        """
