import pandas as pd
import csv
from datetime import datetime

class EquipmentQuery:
    LOG_FILE = r"C:\Users\AL_hamd\OneDrive\Desktop\Chatbot\Config\low_confidence_log.csv"

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data["Equipment Code"] = self.data["Equipment Code"].str.upper().str.strip()
        self.data["Branch"] = self.data["Branch"].str.lower().str.strip()

    def log_low_confidence(self, query, predicted_intent, confidence):
        if confidence < 0.90:
            with open(self.LOG_FILE, mode="a", newline='', encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    query,
                    predicted_intent,
                    f"{confidence:.2f}"
                ])

    def get_quantity_of_branch(self, equipment_code, branch):
        equipment_code = equipment_code.upper()
        branch = branch.lower()

        filtered = self.data[
            (self.data["Equipment Code"] == equipment_code) & 
            (self.data["Branch"] == branch)
        ]
        total_quantity = filtered["Quantity"].sum()
        if not filtered.empty:
            return {
                "success": True,
                "equipment": equipment_code,
                "branch": branch,
                "quantity": int(total_quantity)
            }
        else:
            return {
                "success": False,
                "equipment": equipment_code,
                "branch": branch,
                "message": f"🤖 I didn't find any {equipment_code} in {branch}. Want to check another branch?"
            }
        
    def get_total_quantity_by_equipment(self):
        total_equipment = self.data.groupby("Equipment Code")["Quantity"].sum().reset_index()
        total_equipment.columns = ["Equipment Code", "Total Quantity"]
        return total_equipment

    def get_total_equipment_only(self, equipment_code):
        equipment_code = equipment_code.upper()
        filtered = self.data[self.data["Equipment Code"] == equipment_code]

        if filtered.empty:
            return "No data found for this equipment."

        total = filtered["Quantity"].sum()
        return f"📦 Total quantity of {equipment_code} across all branches: {total}"
if __name__ == "__main__":
    query_tool = EquipmentQuery("equipments.csv")

    user_query = "how many chairs in Arkan"
    predicted_intent = "ask_the_equipment_in_branch"
    confidence = 0.85

    # Log low confidence
    query_tool.log_low_confidence(user_query, predicted_intent, confidence)

    # Perform lookup
    print(query_tool.get_quantity_of_branch("EQ001", "Arkan"))
