import spacy
import json
from thefuzz import process

# Load your equipment and branch config
with open(r"C:\Users\AL_hamd\OneDrive\Desktop\Chatbot\Config\equipment mapping.json", "r") as file:
    data = json.load(file)

with open(r"C:\Users\AL_hamd\OneDrive\Desktop\Chatbot\Config\Locations.json", "r") as file:
    data2 = json.load(file)

equipment_mapping = {k.lower(): v for k, v in data["examples"].items()}
list_equipments = list(equipment_mapping.keys())
# print(list_equipments)
BRANCH_LIST = [k for k in data2["examples"]]

nlp = spacy.load("en_core_web_sm")
def extract_entities(text):
    entities = {"equipment": None, "branch": None}
    doc = nlp(text)

    # --- Equipment Matching ---

    # 1. Try lemmatized direct match
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in list_equipments:
            entities["equipment"] = equipment_mapping.get(lemma)
            break

    # 2. Fuzzy match using noun chunks (after removing stop words)
    if not entities["equipment"]:
        for chunk in doc.noun_chunks:
            chunk_tokens = [token.text.lower() for token in chunk if not token.is_stop]
            clean_chunk = " ".join(chunk_tokens)
            if clean_chunk:
                match = process.extractOne(clean_chunk, list_equipments, score_cutoff=80)
                if match:
                    entities["equipment"] = equipment_mapping.get(match[0])
                    break

    # 3. Fuzzy match using all non-stop tokens (fallback)
    if not entities["equipment"]:
        token_texts = [token.text.lower() for token in doc if not token.is_stop]
        cleaned_text = " ".join(token_texts)
        match = process.extractOne(cleaned_text, list_equipments, score_cutoff=75)
        if match:
            entities["equipment"] = entities["equipment"] = equipment_mapping.get(match[0])


    # --- Branch Matching (unchanged) ---

    for branch in BRANCH_LIST:
        if branch.lower() in text.lower():
            entities["branch"] = branch
            break

    if not entities["branch"]:
        match = process.extractOne(text, BRANCH_LIST, score_cutoff=75)
        if match:
            entities["branch"] = match[0]

    if not entities["branch"]:
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "LOC"]:
                clean_ent = ent.text.strip(" ?.,")
                entities["branch"] = clean_ent
                break

    return entities
if __name__ == "__main__":
    test_inputs = [
        "Input:what about in branch stroll",
    ]

    for user_text in test_inputs:
        entities = extract_entities(user_text)
        print(f"\nInput: {user_text}")
        for k, v in entities.items():
            print(f"{k.capitalize()}: {v}")
