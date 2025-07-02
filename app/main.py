from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import *
from .utils import *
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

app = FastAPI()

# CORS configuration
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

# model
model_name = "Mhammad2023/bert-finetuned-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)


@app.get("/")
async def root():
    return {"message": "Hello in our async FastAPI app for named entity recognition!"}

@app.post("/token_classification", response_model=List[Entity])
async def token_classification(request: TokenClassificationRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        return_offsets_mapping=True
    )

    offset_mapping = inputs.pop("offset_mapping")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    labels = [model.config.id2label[p] for p in predictions]
    scores = torch.softmax(logits, dim=-1).max(dim=-1).values.squeeze(0).tolist()

    offset_mapping = offset_mapping.squeeze(0).tolist()
    start = [s for s, e in offset_mapping]
    end = [e for s, e in offset_mapping]

    # Group tokens into entities
    entities = group_entities(tokens, labels, scores, start, end, request.text)

    return entities



# @app.post("/token_classification", response_model=TokenClassificationResponse)
# async def token_classification(request: TokenClassificationRequest):
#     inputs = tokenizer(
#         request.text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         return_offsets_mapping=True
#     )

#     offset_mapping = inputs.pop("offset_mapping")  # shape: [1, seq_len, 2]
#     input_ids = inputs["input_ids"]

#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits  # [1, seq_len, num_labels]
#     predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()  # list of ints
#     tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
#     labels = [model.config.id2label[p] for p in predictions]
#     scores = torch.softmax(logits, dim=-1).max(dim=-1).values.squeeze(0).tolist()

#     # Convert offsets to character spans
#     offset_mapping = offset_mapping.squeeze(0).tolist()
#     start = [start for start, end in offset_mapping]
#     end = [end for start, end in offset_mapping]

#     return TokenClassificationResponse(
#         text=request.text,
#         tokens=tokens,
#         labels=labels,
#         scores=scores,
#         start=start,
#         end=end
#     )

