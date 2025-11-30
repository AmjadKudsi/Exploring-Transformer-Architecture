#!/bin/sh
# This script will be run when the environment is initialized.
# Add any setup logic here.

echo "Setting up environment…"

if [ -f questions.json ]; then
    wget https://github.com/CodeSignal/learn_quiz-task/archive/refs/tags/v0.9.tar.gz
    tar xvf v*
    mv learn_quiz-task* learn_quiz-task
    rm v*
     
    cp questions.json learn_quiz-task/questions.json

    cd learn_quiz-task
    touch /tmp/.setup_finished
    python3 server.py

    exit 0
fi

source /bootstrap-apps/.virtualenvs/torch/bin/activate

if grep -qF "plt.show()" "src/main.py" || [ -f "src/visualization.py" ]; then
  touch /tmp/.enable_viz
  cd .setup 
  nohup python -m http.server 3000 --bind 0.0.0.0 &
fi

echo "Environment setup complete!"
touch /tmp/.setup_finished