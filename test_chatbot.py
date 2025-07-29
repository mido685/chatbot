import unittest
from extraction import extract_entities
from func_equip import EquipmentQuery
from classifier_pro import IntentClassifierONNX
from main import *
def generate_response(intent, equipment_code, branch):
    if intent == "ask_the_equipment_in_branch":
        if equipment_code and branch:
            result = query_tool.get_quantity_of_branch(equipment_code, branch)
            if not result["success"]:
                return result["message"]
            return random.choice(RESPONSE_TEMPLATES["get_quantity"]).format(
                equipment=result["equipment"],
                branch=result["branch"],
                count=result["quantity"]
            )

# 🔍 Load the classifier and query tool
intent_clf = IntentClassifierONNX(
    'models/logistic_classifier.joblib',
    'models/label_encoder2.joblib',
    r'C:\Users\AL_hamd\OneDrive\Desktop\Chatbot\onnx_model\model.onnx',
    'sentence-transformers/paraphrase-MiniLM-L3-v2'
)
query_tool = EquipmentQuery("equipments.csv")

class TestChatbot(unittest.TestCase):

    def test_entity_extraction(self):
        result = extract_entities("How many chairs in Drive")
        self.assertEqual(result['equipment'], "EQ001")
        self.assertEqual(result['branch'], "Drive")

    def test_branch_quantity_found(self):
        response = generate_response('ask_the_equipment_in_branch',"EQ001", "drive")
        self.assertIn("Total quantity of EQ001", response)

    def test_branch_quantity_not_found(self):
        response = generate_response('ask_the_equipment_in_branch',"EQ001", "unknown_branch")
        self.assertIn("i didn’t find", response)

    def test_total_equipment_only(self):
        response = query_tool.get_total_equipment_only("EQ001")
        self.assertIn("Total quantity of EQ001", response)

    def test_intent_classification(self):
        input_text = "how many chairs in Golf"
        intent, confidence = intent_clf.predict_intent(input_text)
        self.assertEqual(intent, "ask_the_equipment_in_branch")
        self.assertGreaterEqual(confidence, 0.7)

if __name__ == "__main__":
    unittest.main()
