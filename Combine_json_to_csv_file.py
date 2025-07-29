import json
import csv
import os

# Folder where your intent JSON files are stored
json_folder = 'Config/'
output_csv = 'combined_intents.csv'

all_rows = []

# Loop through all JSON files in the folder
for filename in os.listdir(json_folder):
    if filename in ['Locations.json', 'equipment mapping.json','responses.json','ask_all_equipment_totals.json']:
            continue
    if filename.endswith('.json'):
        filepath = os.path.join(json_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            label = data.get("label")
            examples = data.get("examples", [])
            for example in examples:
                all_rows.append({"label": label, "text": example})

# Write to a CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['label', 'text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_rows:
        writer.writerow(row)

print(f"✅ Combined CSV saved to {output_csv}")
