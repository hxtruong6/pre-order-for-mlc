#!/bin/bash

results_dir="results/20250331"

if [ ! -d "$results_dir" ]; then
    mkdir $results_dir
fi

python main2.py --dataset "chd_49" --results_dir $results_dir
python evaluation_test.py --dataset "chd_49" --results_dir $results_dir

python main2.py --dataset "emotions" --results_dir $results_dir
python evaluation_test.py --dataset "emotions" --results_dir $results_dir

python main2.py --dataset "VirusPseAAC" --results_dir $results_dir
python evaluation_test.py --dataset "VirusPseAAC" --results_dir $results_dir

python main2.py --dataset "water-quality" --results_dir $results_dir
python evaluation_test.py --dataset "water-quality" --results_dir $results_dir

python main2.py --dataset "GpositivePseAAC" --results_dir $results_dir
python evaluation_test.py --dataset "GpositivePseAAC" --results_dir $results_dir

python main2.py --dataset "PlantPseAAC" --results_dir $results_dir
python evaluation_test.py --dataset "PlantPseAAC" --results_dir $results_dir

python main2.py --dataset "yeast" --results_dir $results_dir
python evaluation_test.py --dataset "yeast" --results_dir $results_dir

python main2.py --dataset "scene" --results_dir $results_dir
python evaluation_test.py --dataset "scene" --results_dir $results_dir

python main2.py --dataset "HumanPseAAC" --results_dir $results_dir
python evaluation_test.py --dataset "HumanPseAAC" --results_dir $results_dir
