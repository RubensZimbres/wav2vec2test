---
language: pt
datasets:
- common_voice 
metrics:
- wer
tags:
- audio
- speech
- wav2vec2
- pt
- apache-2.0
- portuguese-speech-corpus
- automatic-speech-recognition
- speech
- xlsr-fine-tuning-week
- PyTorch
license: apache-2.0
model-index:
- name: Rubens XLSR Wav2Vec2 Large 53 Portuguese
  results:
  - task: 
      name: Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice pt
      type: common_voice
      args: pt
    metrics:
       - name: Test WER
         type: wer
         value: 20.41%
---


# Wav2Vec2-Large-XLSR-53-Portuguese

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Portuguese using the [Common Voice](https://huggingface.co/datasets/common_voice) dataset.

## Usage

The model can be used directly (without a language model) as follows:

```python
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

test_dataset = load_dataset("common_voice", "pt", split="test[:2%]") 

processor = Wav2Vec2Processor.from_pretrained("Rubens/Wav2Vec2-Large-XLSR-53-Portuguese")
model = Wav2Vec2ForCTC.from_pretrained("Rubens/Wav2Vec2-Large-XLSR-53-Portuguese")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
	speech_array, sampling_rate = torchaudio.load(batch["path"])
	batch["speech"] = resampler(speech_array).squeeze().numpy()
	return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
	logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])
```


## Evaluation

The model can be evaluated as follows on the Portuguese test data of Common Voice.


```python
import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

test_dataset = load_dataset("common_voice", "pt", split="test")
wer = load_metric("wer")

processor = Wav2Vec2Processor.from_pretrained("Rubens/Wav2Vec2-Large-XLSR-53-Portuguese")
model = Wav2Vec2ForCTC.from_pretrained("Rubens/Wav2Vec2-Large-XLSR-53-Portuguese")
model.to("cuda")

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“]'  # TODO: adapt this list to include all special characters you removed from the data
resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
	batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
	speech_array, sampling_rate = torchaudio.load(batch["path"])
	batch["speech"] = resampler(speech_array).squeeze().numpy()
	return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
	inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

	with torch.no_grad():
		logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

  pred_ids = torch.argmax(logits, dim=-1)
	batch["pred_strings"] = processor.batch_decode(pred_ids)
	return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))
```

**Test Result**: 20.41 %


## Training

The Common Voice `train`, `validation` datasets were used for training.

The script used for training can be found at: https://github.com/RubensZimbres/wav2vec2/blob/main/fine-tuning.py
