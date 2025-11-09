# Copyright 2025 [Visual Spatial Tuning] Authors

SYS_PROMPT = "You are a helpful assistant."

THINK_SYSTEM_PROMPT = "You are a helpful assistant. You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here."

DUMMY_CONV = [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "I am VLM-3D, a vision language model trained by researchers from Bytedance Seed."
      },
      {
        "from": "human",
        "value": "Have a nice day!"
      },
      {
        "from": "gpt",
        "value": "You too!"
      }
    ]
