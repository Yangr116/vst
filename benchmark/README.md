# Evaluation
We use the VLMEvalKit to evalute the model on multi-modal benchmarks.

* Install VLMEvalKit
```
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

* Eval on open-source multi-modal benchmarks
Change model name, model_path, datasets in `benchmark/config.json`
Then, just use `auto_run.sh`
```
bash auto_run.sh
```
