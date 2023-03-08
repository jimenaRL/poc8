#!/bin/bash

declare -a CountryNames=(
    'ChileOwn' 'FranceOwn' 'GermanyOwn' 'ItalyOwn' 'Spain' 'UKOwn')

# Iterate the string array using for loop
for val in ${CountryNames[@]}; do
    echo "==================== $val ========================================"
    python emb.py $val
done