import asyncio
import sys
import logging
import json

from src.pipeline.dllm_r0 import DLLMR0
from run import ensure_servers

# Setup logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

async def main():
    print("Starting LLM servers if offline...")
    ensure_servers()
    
    engine = DLLMR0()
    text = """62yo male. CC: Severe chest pain radiating to left arm for 45 minutes HPI: Hyperlipidemia, Smoking 20 pack-years, Family history of CAD (father MI at 55) Symptoms: Crushing substernal chest pain, Pain radiating to left arm and jaw, Diaphoresis, Nausea and vomiting, Shortness of breath Vitals: temperature: 37.2, heart_rate: 110, respiratory_rate: 22, blood_pressure: 90/60, spo2: 94.0"""
    
    res = await engine.analyze(text)
    
    # Save the raw L1 output to see what the model actually produced
    output_data = {
        "entities": res.entities,
        "complexity": res.complexity,
        "urgency": res.urgency,
        "red_flags": res.red_flags,
        "suggested_differentials": res.suggested_differentials,
        "thinking_summary": res.thinking_summary,
    }
    
    with open('swarm_verify.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print("Verification complete. Results saved to swarm_verify.json")

if __name__ == "__main__":
    asyncio.run(main())
