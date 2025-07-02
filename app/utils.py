
def group_entities(tokens, labels, scores, starts, ends, text):
    entities = []
    current_entity = None

    for i, label in enumerate(labels):
        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        tag, entity_type = label.split("-")

        if tag == "B" or current_entity is None or current_entity["entity_group"] != entity_type:
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "entity_group": entity_type,
                "score": scores[i],
                "word": text[starts[i]:ends[i]],
                "start": starts[i],
                "end": ends[i]
            }
        else:  # tag == "I"
            current_entity["score"] = (current_entity["score"] + scores[i]) / 2
            current_entity["word"] += "" + text[starts[i]:ends[i]]
            current_entity["end"] = ends[i]

    if current_entity:
        entities.append(current_entity)

    # Round scores and clean up
    for ent in entities:
        ent["score"] = round(ent["score"], 4)
        ent["word"] = ent["word"].strip()

    return entities
