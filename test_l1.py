import asyncio
import json
from src.pipeline.dllm_r0 import L1_SYSTEM
from src.llm.llama_cpp_client import DLLMClient

async def test_l1():
    client = DLLMClient(base_url="http://127.0.0.1:8081")
    text = """72yo male. CC: Sudden onset right-sided weakness and difficulty speaking for 90 minutes HPI: Atrial fibrillation (not on anticoagulation), hypertension, hyperlipidemia, ex-smoker (40 pack-years) Symptoms: Right hemiparesis (arm > leg), Expressive aphasia — unable to form complete sentences, Right facial droop, Visual neglect on the right, No headache or vomiting Vitals: temperature: 36.8, heart_rate: 88, respiratory_rate: 18, blood_pressure: 185/100, spo2: 96.0 Labs: blood_glucose: 145 mg/dL, INR: 1.0, platelets: 220 x10^9/L, creatinine: 1.1 mg/dL, CT_head: No acute hemorrhage, early hypodensity left MCA territory, NIHSS: 14"""
    
    messages = [
        {"role": "system", "content": L1_SYSTEM},
        {"role": "user", "content": text}
    ]
    resp = client.chat(messages, temperature=0.1, max_tokens=600)
    print("----- RAW OUTPUT -----")
    print(repr(resp.raw))
    print("----------------------")

if __name__ == "__main__":
    asyncio.run(test_l1())
