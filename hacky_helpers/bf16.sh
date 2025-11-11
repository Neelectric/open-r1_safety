python -c "
import json
# config_path = '/gpfs/projects/ehpc283/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/config.json'
config_path = '/home/user/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)
    config['torch_dtype'] = 'bfloat16'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        print('Updated config.json to bfloat16')
        "