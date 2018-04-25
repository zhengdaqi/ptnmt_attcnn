# wpynmt

pytorch nmt containing rnnsearch and transfomer

need to improve ...

# translate

python wtrans.py -m model_file -i 900

# evaluate alignment

score-alignments.py -d path/900 -s zh -t en -g wa -i force_decoding_alignment
