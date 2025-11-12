**CENG467 Proje 1 – Uygulama Planı (CoMAT, Shapley, GRPO)**

- Sürüm: v1.0
- Son güncelleme: 2025-11-10
- Kapsam: Bu plan, CENG467_Project1.pdf’teki gereksinimler ile mevcut kod şablonundaki STUB bölgelerini tamamlayarak tüm deneyleri çalıştırma ve rapora veri üretme adımlarını uçtan uca kapsar.

**İçindekiler**
- Hazırlık ve Ortam
- Kod Tamamlama (STUB’lar)
- Değerlendirme Deneyleri (Q2.b–Q2.g)
- Shapley Analizi (Q2.h–Q2.j)
- GRPO Finetuning (Q4)
- Rapor ve Teslim

---

**Hazırlık ve Ortam**
- Python ve paketler
  - Python 3.12 önerilir. PyTorch’u cihazınıza uygun şekilde kurun (GPU varsa CUDA’lı sürüm).
  - Gerekli paketler: `pip install -r requirements.txt`
  - GPU doğrulama: Python içinde `import torch; torch.cuda.is_available()` True dönmeli (varsa).
- Donanım/Runtime seçenekleri
  - Yerel makine yetersizse Google Colab GPU (T4) ile çalışın. Rehber: PDF’te verilen Colab makalesi.
  - Colab’da çalışma için proje klasörünü Drive’a taşıyıp mount edin; kodu parça parça kopyalamayın.
- Depo düzeni (ilgili dosyalar)
  - `main.py`: Deney koşum dosyası (dataset/model/parametreler buradan geçer)
  - `mmlu_redux.py`: MMLU-Redux soru akışı ve regex tabanlı cevap çıkarımı
  - `utils.py`: Model çağrısı (`predict_model`) – STUB
  - `CoMAT_Instruction.py`: Talimat metni (prompt) – STUB
  - `shapley_value_evaluation.py`: Shapley değeri analizi – STUB’lar
  - `grpo_finetune.py`: GRPO ile basit RLFT akışı – STUB’lar ve örnek senaryo
  - `prompt-instruction.txt`: CoMAT prompt içeriği (Q2 için talimat)
  - `mmlu-redux-college_mathematics_dataset.csv`: Değerlendirme veri kümesi
  - `evaluation_with_steps.csv`: Shapley analizi için oyuncu (adım) kombinasyonları ve doğruluk 0/1 veri seti

---

**Kod Tamamlama (STUB’lar)**
- 1) `CoMAT_Instruction.py` (Q2.a.II – Kolay)
  - Görev: `INSTRUCTION` adlı değişkeni `prompt-instruction.txt` içeriğiyle doldur.
  - Öneri: Basitçe dosyadan okuyup bir string’e ata veya içerik sabitini doğrudan `INSTRUCTION = "..."` olarak yerleştir.
  - Not: Bu dosyada yalnızca talimatı ekleyin; başka değişiklik yapmayın.

- 2) `utils.py::predict_model` (Q2.a.I)
  - Girdi: `messages` listesi (role/content), `configuration = {temperature, max_token_limit}`
  - Model: QWEN2/QWEN3 Instruct sürümleri (chat-template destekli)
  - Uygulama adımları:
    - `tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)` ile tek string prompt veya tensor üret.
    - Cihaz yerleştirme: tensörleri `cuda` (varsa) veya `cpu`’ya taşı.
    - `model.generate` çağrısı:
      - `max_new_tokens=configuration["max_token_limit"]`
      - `temperature=configuration["temperature"]`
      - `do_sample = (temperature > 0)`
      - İsteğe bağlı: `eos_token_id=tokenizer.eos_token_id` (varsa)
    - `tokenizer.decode(..., skip_special_tokens=True)` ile çıktı string’i döndür.
  - Notlar:
    - Kodun geri kalanı (main/mmlu_redux) doğrudan bu fonksiyona dayanır; isim/parametreleri değiştirmeyin.

- 3) `shapley_value_evaluation.py` (Q2.a.III)
  - Adımlar ve beklenen mantık:
    - `get_missing_steps(row, steps)`: CSV’deki `step{i}_missing` sütunlarına bakıp 1 olanları toplayın ve `tuple(sorted(...))` olarak döndürün. Örn: `row['step2_missing'] == 1` ise 2 eklenecek.
    - `generate_all_subsets(steps)`: Mevcut (hazır).
    - `compute_v_S(df, all_subsets_missing)`: Her `S` (eksik adımlar kümesi) için `df[df['missing_steps'] == S]['is_correct'].mean()` ile `v(S)` hesaplayın. Kayıt yoksa `np.nan` atayın.
    - `compute_marginal_contributions(steps, v_S)`: Tüm permütasyonlar için her adım i’nin marjinal katkısı: `Δ_i(π) = v(S_i) - v(S_i ∪ {i})` (burada S_i, i’den önce gelen “eksik adımlar” kümesidir). `nan` içeren durumlarda permütasyonu geçersiz sayın.
    - `compute_shapley_values(Delta_sum, valid_permutations_count, steps)`: `ϕ_i = Delta_sum[i] / valid_permutations_count`.
    - `main()`:
      - CSV’yi oku → `missing_steps` kolonu üret → tüm alt kümeleri üret → `v(S)` sözlüğünü hesapla → marjinal katkılar ve Shapley değerleri → sonuçları yazdır.
  - Not: Kod şablonu “eksik adımlar” (missing) üzerinden tanımlar yapıyor; bu yüzden marjinal katkı formülü `v(S) - v(S∪{i})` olmalıdır (adımı kaldırmanın etkisi).

- 4) `grpo_finetune.py` (Q4.a)
  - `reward_function(...)`:
    - Her completion için son 20 karakterdeki son A/B/C/D eşleşmesini regex ile çıkarın (şablonda hazır fonksiyon var).
    - `correct_answer` ile karşılaştırıp eşleşme durumunda 1.0, aksi 0.0 döndürün.
    - Girdi şekilleri listelidir (8 tekrar); vektörize veya döngü ile liste üretin.
  - `preprocess_function(examples)`:
    - Girdi: HF `datasets` batch’i (ör. listeler).
    - Çıktı kolonları:
      - `prompt`: `INSTRUCTION + "\n\n-----------\n\nQuestion: ...\n\nOptions:\n..."` formatında; `choices` string listesini `\n` ile birleştir. (A., B., C., D. ön ekleri yoksa ekleyin.)
      - `correct_answer`: `answer` alanını 0–3 → `A/B/C/D` harfine dönüştürüp yazın.
      - `question` ve `choices` korunabilir; `remove_columns=['answer']` çağrısı şablonda mevcut.
  - Doğrulama: Dosyadaki örnek senaryoda `rewards = [1.0]*8` ürettiğini kontrol edin.

---

**Değerlendirme Deneyleri (Q2.b–Q2.g)**
- 1) Kurulum ve temel çalıştırma
  - Komut sablonu:
    - `python main.py --dataset mmlu-redux-college_mathematics --method comat --model qwen2 --temperature 0.1 --max_token_limit 2000`
  - Çıktılar:
    - Sonuç JSON: `final_results/mmlu-redux-college_mathematics/comat/qwen2/comat_qwen2.json`
    - Log: `final_results/mmlu-redux-college_mathematics/comat/qwen2/comat_qwen2_log.txt`
  - Not: Tüm 100 örneği çalıştırmak uzun sürebilir; kısmi çalıştırma kabul. Regex kaynaklı `final_answer=-1` durumlarını manuel kontrol ederek doğruluğu (accuracy) elle düzeltin.

- 2) Deney matrisi
  - Q2.b: `qwen2`, `temperature=0.1`, `max_token_limit=2000`
  - Q2.c: `qwen2`, `temperature=0.7`, `max_token_limit=2000`
  - Q2.f: `qwen3`, `temperature=0.1`, `max_token_limit=2000` (tam set şart değil)
  - Q2.g: `qwen3`, `temperature=0.1`, `max_token_limit=4000`
  - Her koşul için: Çalışma süresi, gözlemlenen kalite, JSON’da doğru/yanlış sayısı ve regex hataları not edilecek.

- 3) Yorumlama (rapora konulacak)
  - Q2.d: Sıcaklığın (temperature) etkisi: 0.7 → çeşitlilik ve tutarlılık dengesi → doğruluğa etkiler.
  - Q2.e: CoMAT’ın beklenenin altında kalma olasılıkları: model kapasitesi, talimat uyumu, maksimum token uzunluğu, prompt formatı, regex ölçüm hataları, veri zorluğu vb.
  - Q2.f–Q2.g: QWEN2 ↔ QWEN3 karşılaştırması ve `max_token_limit` yükseltiminin etkisi.

---

**Shapley Analizi (Q2.h–Q2.j)**
- 1) Çalıştırma
  - `python shapley_value_evaluation.py`
  - Kod, `evaluation_with_steps.csv`’den veri okuyacak, “eksik adımlar”a göre `v(S)` çıkaracak, tüm permütasyonlar üzerinden marjinal katkıları toplayıp Shapley değerlerini yazdıracak.
- 2) Raporlama
  - Q2.h: s1–s4 için ϕ değerlerini tabloleyin; negatif katkıların mümkün olduğunu belirtin.
  - Q2.i: En yüksek Shapley alan adımı ve olası nedenleri.
  - Q2.j: Appendix D bulgusuna atıfla (s1 ve s2 birlikte kaldırma vs yalnız s1) bağımlılık yorumu.

---

**GRPO Finetuning (Q4)**
- 1) STUB’ları tamamlayın ve örnek senaryoyu doğrulayın
  - `reward_function` çalışınca oyuncak örnekte `[1.0, ..., 1.0]` üretmeli.
  - `preprocess_function`: `prompt` ve `correct_answer` üretimi doğru olmalı.
- 2) Veri hazırlığı ve eğitim
  - Kod veriyi 80/20 böler, baz çıktıları üretir, ardından GRPO ile eğitir.
  - Model/Tokenizer: `Qwen/Qwen2-0.5B-Instruct` (CUDA varsa GPU’da çalıştırın).
- 3) Çıktıları inceleme
  - Eğitim öncesi/sonrası `json` çıktılarından bir veya birkaç örnek seçip kalite değişimini gözlemleyin.
  - Beklenti: Büyük sıçrama değil; küçük iyileşmeler ve tutarlılıkta düzenleme görülebilir.
- 4) Raporlama
  - Q4.a: Seçtiğiniz örnek(ler) için önce/sonra çıktılarını yazın; düşük başarı nedenleri (küçük veri, basit ödül fonksiyonu, kısa eğitim vb.).
  - Q4.b: GRPO’nun göreli karşılaştırma mantığının (8 completion) pratik faydaları.
  - Q4.c: Ödül fonksiyonu geliştirme fikirleri (adım doğruluğu, ara akıl yürütme tutarlılığı, biçim kısıtları, cezalar vb.).

---

**Rapor ve Teslim**
- Rapor (PDF)
  - Tüm yazılı çözümler (Q1 serisi) ve analizler el yazısı olmalı, A4’e yazılıp taranmalı.
  - Deney bulguları: Her koşulun doğruluğu, çalışma süreleri/izlenimler, Shapley sonuçları, GRPO önce/sonra örnekleri.
- Kod Teslimi (ZIP)
  - Tüm STUB’lar doldurulmuş olmalı. Kod dosyalarının başına grup numarası/isimler yorum olarak eklenmeli.
  - Deponun tamamı sıkıştırılıp ilgili adlandırma ile yüklenmeli.

---

**Kontrol Listesi (Özet)**
- [ ] Ortam kuruldu, GPU erişimi doğrulandı.
- [ ] `CoMAT_Instruction.py` → `INSTRUCTION` dolduruldu.
- [ ] `utils.py::predict_model` tamamlandı (temperature, max_new_tokens parametreleri geçirildi).
- [ ] `shapley_value_evaluation.py` STUB’ları tamamlandı ve çalıştı; ϕ değerleri raporlandı.
- [ ] `grpo_finetune.py` → `reward_function` ve `preprocess_function` tamamlandı; örnek senaryo `[1.0]*8` doğrulandı.
- [ ] Deneyler (Q2.b–Q2.g) çalıştı; JSON/log sonuçları ve regex düzeltmeleri kontrol edildi.
- [ ] Rapor (el yazısı) hazırlandı; kod zip paketi oluşturuldu.

