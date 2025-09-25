#!/bin/bash -eu
# NOTE: Ensure to run with `bash script/xx.sh`

META_MODEL="gpt-4.1-2025-04-14"
MODEL="gpt-4.1-2025-04-14"
REASON="gen"
TASKS=( \
	bbh_geometric_shapes \
	bbh_salient_translation_error_detection \
	bbh_disambiguation_qa \
	bbh_movie_recommendation \
	bbh_boolean_expressions \
	bbh_causal_judgement \
	bbh_sports_understanding \
	bbh_word_sorting \
)
SEEDS=(1000 2000 3000)

echo "Running for 8 shot..."
for seed in "${SEEDS[@]}"; do
	echo "  Seed ${seed}"
	for task in "${TASKS[@]}"; do
		uv run src/run_main_api.py \
			--task "${task}" \
			--meta_model "${META_MODEL}" \
			--model "${MODEL}" \
			--reason_type "${REASON}" \
			--prompt_type shot \
			--shot 8 \
			--seed "${seed}" &
	done
	wait
done

echo "Running for 16 shot..."
for seed in "${SEEDS[@]}"; do
	echo "  Seed ${seed}"
	for task in "${TASKS[@]}"; do
		uv run src/run_main_api.py \
			--task "${task}" \
			--meta_model "${META_MODEL}" \
			--model "${MODEL}" \
			--reason_type "${REASON}" \
			--prompt_type shot \
			--shot 16 \
			--seed "${seed}" &
	done
	wait
done

echo "Running for 32 shot..."
for seed in "${SEEDS[@]}"; do
	echo "  Seed ${seed}"
	for task in "${TASKS[@]}"; do
		uv run src/run_main_api.py \
			--task "${task}" \
			--meta_model "${META_MODEL}" \
			--model "${MODEL}" \
			--reason_type "${REASON}" \
			--prompt_type shot \
			--shot 32 \
			--seed "${seed}" &
	done
	wait
done

echo "Running for 64 shot..."
for seed in "${SEEDS[@]}"; do
	echo "  Seed ${seed}"
	for task in "${TASKS[@]}"; do
		uv run src/run_main_api.py \
			--task "${task}" \
			--meta_model "${META_MODEL}" \
			--model "${MODEL}" \
			--reason_type "${REASON}" \
			--prompt_type shot \
			--shot 64 \
			--seed "${seed}" &
	done
	wait
done

echo "Running for n shot..."
for seed in "${SEEDS[@]}"; do
	echo "  Seed ${seed}"
	for task in "${TASKS[@]}"; do
		uv run src/run_main_api.py \
			--task "${task}" \
			--meta_model "${META_MODEL}" \
			--model "${MODEL}" \
			--reason_type "${REASON}" \
			--prompt_type shot \
			--shot 0 \
			--seed "${seed}" &
	done
	wait
done

echo "Running for cheat sheet..."
for seed in "${SEEDS[@]}"; do
	echo "  Seed ${seed}"
	for task in "${TASKS[@]}"; do
		uv run src/run_main_api.py \
			--task "${task}" \
			--meta_model "${META_MODEL}" \
			--model "${MODEL}" \
			--reason_type "${REASON}" \
			--prompt_type cheat \
			--shot 0 \
			--seed "${seed}" &
	done
	wait
done

echo Done