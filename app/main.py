from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .schemas import *
from .utils import *
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from contextlib import asynccontextmanager

# Global model variables
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = "Mhammad2023/bert-finetuned-ner-torch"
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_name)
    app.state.model = AutoModelForTokenClassification.from_pretrained(model_name).to("cpu")
    torch.set_num_threads(1)
    app.state.model.eval()
    yield

app = FastAPI(lifespan=lifespan)
# app = FastAPI()

# CORS configuration
#Good for allowing external frontend apps to interact with your API.
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

# model
# model_name = "Mhammad2023/bert-finetuned-ner-torch"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)

@app.get("/")
async def root():
    return {f"message": "Hello in our async FastAPI app for named entity recognition!"}


@app.post("/token_classification", response_model=List[Entity])
async def token_classification(request: TokenClassificationRequest, req: Request):
    try:
        tokenizer = req.app.state.tokenizer
        model = req.app.state.model

        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]
        predictions = torch.argmax(logits, dim=-1).tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        labels = [model.config.id2label[p] for p in predictions]
        scores = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()

        start = [s for s, _ in offset_mapping]
        end = [e for _, e in offset_mapping]

        entities = group_entities(tokens, labels, scores, start, end, request.text)
        return entities

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/token_classification", response_model=List[Entity])
# async def token_classification(request: TokenClassificationRequest):
#     try:
#         inputs = tokenizer(
#             request.text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#             return_offsets_mapping=True
#         )

#         offset_mapping = inputs.pop("offset_mapping")
#         input_ids = inputs["input_ids"]

#         with torch.no_grad():
#             outputs = model(**inputs)

#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()
#         tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
#         labels = [model.config.id2label[p] for p in predictions]
#         scores = torch.softmax(logits, dim=-1).max(dim=-1).values.squeeze(0).tolist()

#         offset_mapping = offset_mapping.squeeze(0).tolist()
#         start = [s for s, e in offset_mapping]
#         end = [e for s, e in offset_mapping]

#         # Group tokens into entities
#         entities = group_entities(tokens, labels, scores, start, end, request.text)

#         return entities
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




