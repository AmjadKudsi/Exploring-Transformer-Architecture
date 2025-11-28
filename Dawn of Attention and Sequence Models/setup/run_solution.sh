#!/bin/bash

# Check if setup is complete
if [ ! -f "/tmp/.setup_finished" ]; then
    echo "The setup is not finished yet. Try again in a few seconds." >&2
    exit 1
fi


if [ -f questions.json ]; then
    cd learn_quiz-task
    python3 format_answers.py

    exit 0
else
    cd /usercode/FILESYSTEM

    # Activate environment
    source /bootstrap-apps/.virtualenvs/torch/bin/activate
    
    if [ -f /tmp/.enable_viz ]; then
      # Copy to temp dir, substitute plt.show with plt.savefig, execute and remove
      #cp -r src .setup/temp_src && cd .setup/temp_src
      #VIZ_FILE=$(grep -l 'plt.show()' *.py) # Get actual file calling plt.show (it may not be main)
      
      # sed -i 's/plt\.show()[[:space:]]*/plt.savefig("plot.png")/' "$VIZ_FILE"
      # python3 main.py
      # cd .. && mv temp_src/plot.png . && rm -r temp_src
      
      cd src
      export MPLBACKEND=Agg
      python3 -c "exec(open('main.py').read()); import matplotlib.pyplot as plt; plt.savefig('../.setup/plot.png')"
      
    else
      python3 src/main.py 2>&1 | grep -E -v "cuda|tensorflow|WARNING|E0000|xla|cuFFT|cuDNN|cuBLAS|AVX2|Bert"
    fi
fi

