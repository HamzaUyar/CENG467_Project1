**CENG467 Proje 1 – Rapor Taslağı**

- Grup No: …
- Üyeler: …
- Ortam: Python 3.12, PyTorch (GPU varsa CUDA), Transformers, TRL

---

**Q1 – Formalizing math questions**

- Q1.a (2p) – El ile çözüm (LLM kullanma):
  - Soru (MQ1):
    - “218 galon kapasiteli boş bir yakıt tankı önce kısmen A yakıtı ile, sonra kapasiteye kadar B yakıtı ile dolduruluyor. A yakıtı hacmen %12 etanol, B yakıtı %16 etanol içeriyor. Tam dolu tankta 30 galon etanol varsa kaç galon A yakıtı eklenmiştir?”
    - Şıklar: A) 122  B) 150  C) 100  D) 80  E) 50
  - Çözüm yolunun iskeleti (elden yaz):
    - Değişken tanımı: a = A yakıtı galon; (218 − a) = B yakıtı galon.
    - Etanol denklemi: 0.12·a + 0.16·(218 − a) = 30.
    - a’yı çöz; 0 ≤ a ≤ 218 olmalı. Bulduğun değeri en yakın şıkla eşleştir.
  - Kontrol listesi:
    - Birim tutarlılığı (galon), yüzde→ondalık dönüşümü, aralık kontrolü.
    - Denklemi doğru kurduğunu ve toplu etanol miktarını 30’a eşitlediğini doğrula.

- Q1.b (5p) – Dört adımlı biçimselleştirme (elden yaz):
  - 1) Identification and Definition: Değişkenler, sabitler, fonksiyonlar, predicate’ler.
  - 2) Structural Logic Translation: Kurallar ve ilişkiler (seçeneklerden bağımsız).
  - 3) Explicit Factual Representation: Metindeki açık gerçeklerin mantıksal ifadeleri.
  - 4) Question Formalization: Sorunun sembolik formu (seçeneklerden bağımsız).

- Q1.c (3p) – Reasoning Execution & Final Answer (elden yaz):
  - Below is a ready-to-copy English writeup (symbolic first, then optional numeric instantiation). Handwrite it on paper.
  
  Reasoning Execution (symbolic):
  1) From Step 4, we must find a such that (a + b = C) ∧ (p_A · a + p_B · b = E).
  2) Using the capacity relation b = C − a and substituting into the ethanol balance:
     E = p_A · a + p_B · (C − a).
  3) Rearranging gives E = (p_A − p_B) · a + p_B · C, hence
     (p_A − p_B) · a = E − p_B · C.
  4) Solve for a: a = (E − p_B · C) / (p_A − p_B).
  5) Compute b = C − a and verify feasibility: 0 ≤ a ≤ C, 0 ≤ b ≤ C.
  6) Report the numeric value of a (gallons) and only then map it to the matching option.

  Optional numeric instantiation (for verification):
  Let C = 218, p_A = 0.12, p_B = 0.16, E = 30.
  Then a = (E − p_B·C)/(p_A − p_B) = (30 − 0.16·218)/(0.12 − 0.16) = 122,
  b = 218 − 122 = 96, constraints hold. Final Answer (option): A.

- Q1.d (5p) – Yöntem değerlendirme (elden yaz):
  - Mekanik çözüm kolaylaştı mı? Güçlü/zayıf yanlar, sınırlılıklar.

- Q1.e (5p) – Ek adım önerisi (elden yaz):
  - Gerekliyse örnek bir ek adım öner ve gerekçelendir; ya da mevcut sıranın yeterli olduğunu savun.

---

**Q2 – Evaluations & Shapley**

- Çalıştırma komutu (örnek):
  - `python main.py --dataset mmlu-redux-college_mathematics --method comat --model qwen2 --temperature 0.1 --max_token_limit 2000`
  - Çıktılar: `final_results/.../comat_qwen2.json`, `..._log.txt`
  - Not: Regex kaynaklı `final_answer = -1` durumlarını dosyadan kontrol edip elle düzeltin.

- Q2.b (5p) – qwen2, T=0.1, max=2000:
  - Gözlemler: …  Doğruluk: …  Çalışma süresi: …  Notlar: …

- Q2.c (5p) – qwen2, T=0.7, max=2000:
  - Gözlemler: …  Doğruluk: …  Karşılaştırma: …

- Q2.d (5p) – T etkisi yorumu:
  - 0.1 ↔ 0.7 etkisi (çeşitlilik/kararlılık), doğruluk farkı gerekçesi.

- Q2.e (5p) – Olası underperform nedenleri:
  - Model kapasitesi, prompt uyumu, token limiti, ölçüm (regex), veri zorluğu vb.

- Q2.f (5p) – qwen3, T=0.1, max=2000:
  - Gözlemler, süre ve sonuçlar; qwen2 ile karşılaştırma.

- Q2.g (5p) – qwen3, T=0.1, max=4000:
  - Değişim var mı? Olası nedenler.

- Q2.h–Q2.j – Shapley (çıktıları buraya yaz):
  - s1, s2, s3, s4 için ϕ değerleri: …
  - En yüksek ϕ ve gerekçe: …  Bağımlılık yorumu (Appendix D): …

---

**Q3 – CoMAT başarısız örnek analizi**

- Qf: (qwen2 başarısız, qwen3 başarılı olan bir örnek bul)
- Q3.a: LLM adımları (yanlış çözüm) ve varsa qwen3 doğru çözümü.
- Q3.b: Nerede hata yaptı? Adım analizi.
- Q3.c: Düzeltici ek adım önerisi.

---

**Q4 – GRPO Finetuning**

- Q4.a – STUB’lar tamamlandıktan sonra:
  - Oyuncak örnek ödül çıktısı: `[1.0, …, 1.0]` (8 adet) doğrulandı mı? Evet/Hayır
  - Öncesi/Sonrası json’lardan seçili soru: …  Önce: …  Sonra: …  Gözlem: …
  - Neden beklenen seviyede değil? (küçük veri, basit ödül, kısa eğitim, vb.)

- Q4.b – İyileşme örnekleri:
  - …

- Q4.c – Daha iyi ödül önerileri:
  - Adım-temelli kısmi puan, format cezaları, ara akıl yürütme tutarlılığı vb.

---

**Ekler**

- Komutlar: çalıştırma parametreleri ve tarih/saatler.
- Ortam ve sürümler: `pip freeze` özeti, cihaz bilgisi.
- Notlar: Regex düzeltmeleri, zaman planlaması.
