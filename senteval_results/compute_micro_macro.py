import json

with open("senteval_results_all.json", "r") as file:
    json_data = json.load(file)

model_names = ["me", "lstme", "blstme", "blstmpme"]
models = []

for model_data in json_data:
    macro = 0
    micro = 0
    task_count = 0
    sample_count = 0
    for task, values in model_data.items():
        macro += values["devacc"]
        task_count += 1
        micro += values["devacc"] * values["ndev"]
        sample_count += values["ndev"]

    models.append({"micro": micro / sample_count, "macro": macro / task_count})

for scores, name in zip(models, model_names):
    print(f"Model {name} scores: {scores}")
