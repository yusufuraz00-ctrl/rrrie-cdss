import asyncio
import sys
import logging
import json

from src.pipeline.dllm_r0 import DLLMR0
from run import ensure_servers

# Setup logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

async def main():
    print("Starting LLM servers if offline...")
    ensure_servers()
    
    engine = DLLMR0()
    text = '''Hastanın İfadesi: "Sabah yataktan kalkarken belim resmen kilitlendi abi. Eğilip çorabımı bile giyemedim. Öyle ağır bir şey de kaldırmadım aslında, dün akşam balkonda terli terli cereyanda oturmuştum ondan mı oldu bilmem. Şimdi sağa sola dönerken bıçak saplanıyor sanki, nefesimi kesiyor ağrısı. Bacağıma vuran bir sızı yok, uyuşma falan da yok ama tahta gibi tutuldum kaldım yani, doğrulurken canımdan can gidiyor."
    
    Triyaj Notu (Gözlem): 35 Yaş, Erkek. Ateş: 36.5°C, Nabız: 78/dk, Tansiyon: 120/80 mmHg. Fizik muayenede lomber bölgede (bel) kas spazmı mevcut. Bacaklarda his kaybı, güç kaybı veya refleks eksikliği yok (Nörolojik muayene normal).'''
    
    res = await engine.analyze(text)
    
    print("\n" + "="*50)
    print("FINAL ENTITIES (JSON):")
    print(json.dumps(res.entities, indent=2, ensure_ascii=False))
    print("\nCOMPLEXITY:", res.complexity)
    print("URGENCY:", res.urgency)
    print("RED FLAGS:", res.red_flags)
    print("DIFFERENTIALS:", res.suggested_differentials)
    print("\nTHINKING TRACE:")
    print(res.thinking_summary)

if __name__ == "__main__":
    asyncio.run(main())
