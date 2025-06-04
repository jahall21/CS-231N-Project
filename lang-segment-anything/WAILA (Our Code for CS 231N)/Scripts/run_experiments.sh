#!/bin/bash
python waila.py --n_terms 10 --n_samples 5 --word_model "conceptnet_similar_words.json" --root_output_folder "./output_masks_conceptnet"
python waila.py --n_terms 10 --n_samples 5 --word_model "w2v_similar_words.json" --root_output_folder "./output_masks_w2v"

