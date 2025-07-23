from tinyllama_training import SmartHomeInference

# Test the trained model
inference = SmartHomeInference()

# Test cases
test_cases = [
    "07:00 | bedroom_light | turn_on [SEP]",
    "08:00 | kitchen_light | turn_on [SEP] 08:05 | coffee_maker | start [SEP]",
    "19:00 | living_room_light | turn_on [SEP] 19:05 | tv | turn_on [SEP]"
]

for test_input in test_cases:
    predicted = inference.predict_next_action(test_input, max_new_tokens=30)
    print(f"\nInput: {test_input}")
    print(f"Predicted: {predicted}")