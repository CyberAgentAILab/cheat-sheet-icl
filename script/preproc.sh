#!/bin/bash -eu
# NOTE: Ensure to run with `bash script/xx.sh`

META_MODEL="gpt-4.1-2025-04-14"
REASON="gen"

echo "Starting rationale augmentation..."
uv run src/generate_reason_api.py --task bbh_disambiguation_qa --meta_model ${META_MODEL} &
uv run src/generate_reason_api.py --task bbh_geometric_shapes --meta_model ${META_MODEL} &
uv run src/generate_reason_api.py --task bbh_movie_recommendation --meta_model ${META_MODEL} &
uv run src/generate_reason_api.py --task bbh_salient_translation_error_detection --meta_model ${META_MODEL}
wait
uv run src/generate_reason_api.py --task bbh_boolean_expressions --meta_model ${META_MODEL} &
uv run src/generate_reason_api.py --task bbh_causal_judgement --meta_model ${META_MODEL} &
uv run src/generate_reason_api.py --task bbh_sports_understanding --meta_model ${META_MODEL} &
uv run src/generate_reason_api.py --task bbh_word_sorting --meta_model ${META_MODEL}
wait

echo "Creating cheat sheets..."
uv run src/cheatsheet_api.py --task bbh_geometric_shapes --meta_model ${META_MODEL} --reason_type ${REASON} --seed 1000 &
uv run src/cheatsheet_api.py --task bbh_salient_translation_error_detection --meta_model ${META_MODEL} --reason_type ${REASON} --seed 1000 &
uv run src/cheatsheet_api.py --task bbh_disambiguation_qa --meta_model ${META_MODEL} --reason_type ${REASON} --seed 1000 &
uv run src/cheatsheet_api.py --task bbh_movie_recommendation --meta_model ${META_MODEL} --reason_type ${REASON} --seed 1000 &
uv run src/cheatsheet_api.py --task bbh_boolean_expressions --meta_model ${META_MODEL} --reason_type ${REASON} --seed 1000 &
uv run src/cheatsheet_api.py --task bbh_causal_judgement --meta_model ${META_MODEL} --reason_type ${REASON} --seed 1000 &
uv run src/cheatsheet_api.py --task bbh_sports_understanding --meta_model ${META_MODEL} --reason_type ${REASON} --seed 1000 &
uv run src/cheatsheet_api.py --task bbh_word_sorting --meta_model ${META_MODEL} --reason_type ${REASON} --seed 1000
wait
uv run src/cheatsheet_api.py --task bbh_geometric_shapes --meta_model ${META_MODEL} --reason_type ${REASON} --seed 2000 &
uv run src/cheatsheet_api.py --task bbh_salient_translation_error_detection --meta_model ${META_MODEL} --reason_type ${REASON} --seed 2000 &
uv run src/cheatsheet_api.py --task bbh_disambiguation_qa --meta_model ${META_MODEL} --reason_type ${REASON} --seed 2000 &
uv run src/cheatsheet_api.py --task bbh_movie_recommendation --meta_model ${META_MODEL} --reason_type ${REASON} --seed 2000 &
uv run src/cheatsheet_api.py --task bbh_boolean_expressions --meta_model ${META_MODEL} --reason_type ${REASON} --seed 2000 &
uv run src/cheatsheet_api.py --task bbh_causal_judgement --meta_model ${META_MODEL} --reason_type ${REASON} --seed 2000 &
uv run src/cheatsheet_api.py --task bbh_sports_understanding --meta_model ${META_MODEL} --reason_type ${REASON} --seed 2000 &
uv run src/cheatsheet_api.py --task bbh_word_sorting --meta_model ${META_MODEL} --reason_type ${REASON} --seed 2000
wait
uv run src/cheatsheet_api.py --task bbh_geometric_shapes --meta_model ${META_MODEL} --reason_type ${REASON} --seed 3000 &
uv run src/cheatsheet_api.py --task bbh_salient_translation_error_detection --meta_model ${META_MODEL} --reason_type ${REASON} --seed 3000 &
uv run src/cheatsheet_api.py --task bbh_disambiguation_qa --meta_model ${META_MODEL} --reason_type ${REASON} --seed 3000 &
uv run src/cheatsheet_api.py --task bbh_movie_recommendation --meta_model ${META_MODEL} --reason_type ${REASON} --seed 3000 &
uv run src/cheatsheet_api.py --task bbh_boolean_expressions --meta_model ${META_MODEL} --reason_type ${REASON} --seed 3000 &
uv run src/cheatsheet_api.py --task bbh_causal_judgement --meta_model ${META_MODEL} --reason_type ${REASON} --seed 3000 &
uv run src/cheatsheet_api.py --task bbh_sports_understanding --meta_model ${META_MODEL} --reason_type ${REASON} --seed 3000 &
uv run src/cheatsheet_api.py --task bbh_word_sorting --meta_model ${META_MODEL} --reason_type ${REASON} --seed 3000
wait

echo Done