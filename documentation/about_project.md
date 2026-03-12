# [cite_start]A Traceable Retrieval-Based Clinical Decision Support System for Breast Cancer Histopathology [cite: 8]

[cite_start]**Kurum:** Istanbul Technical University Faculty of Computer and Informatics [cite: 1, 3] [cite_start]/ Department of Artificial Intelligence and Data Engineering [cite: 2, 4]
[cite_start]**Ders:** Artifical Intelligence & Data Engineering Design I [cite: 5]
[cite_start]**Danışman:** Prof. Dr. Behçet Uğur Töreyin [cite: 10, 11]
[cite_start]**Geliştiriciler:** Emir Arda Eker (150220331) & Faruk Rıza Öz (150210753) [cite: 9]
[cite_start]**Tarih:** Jan, 2026 [cite: 12]

---

## Proje Özeti (Summary)
* [cite_start]Bu proje, Tüm Slayt Görüntüleri (WSI) üzerinden meme kanseri histopatolojisinin analizi için izlenebilir bir klinik karar destek sistemi geliştirmeyi amaçlamaktadır[cite: 26, 35].
* [cite_start]Sistem, otonom tıbbi kararlar vermek yerine, klinik kullanıcılara morfolojik olarak benzer referans vakaları sunarak ve kanıta dayalı yapılandırılmış bir teşhis özeti üreterek destek olacak şekilde tasarlanmıştır[cite: 27, 40].
* [cite_start]Temel hedef; dijital patoloji iş akışlarında her bir tanısal ifadenin, görsel girdi bölgeleri ve karşılık gelen referans raporlarla doğrulanabilmesini sağlayarak izlenebilirliği, güvenliği ve kullanılabilirliği artırmaktır[cite: 28].

## Mimari ve Modüller (Proposed Solution)
* [cite_start]Sistem, yeni bir temel model eğitmek yerine, mevcut patoloji modellerini verimli uzun-dizi işleme ve yapılandırılmış klinik akıl yürütme yetenekleriyle tek bir boru hattında birleştirir[cite: 61].
* [cite_start]**WSI Ingestion & Preprocessing:** WSI dosyalarından yama (patch) düzeyinde özellikler çıkarılır[cite: 63, 173].
* [cite_start]**Patch Embedding:** Çıkarılan yamalar, dondurulmuş (frozen) bir patoloji temel modeli (örn. UNI) kullanılarak sabit uzunluklu vektörlere dönüştürülür[cite: 63, 175, 290].
* [cite_start]**Sequence Model (SAMBA / Mamba):** Çok uzun yama dizileri, tek GPU kısıtlamaları altında çalışabilmek için doğrusal karmaşıklığa sahip bir dizi modeli ile slayt seviyesinde temsillere dönüştürülür[cite: 64, 153, 154].
* [cite_start]**CMEA (Case Retrieval):** Morfolojik olarak benzer referans vakalar ve bunlarla ilişkili teşhis raporları veri tabanından (GDC, Quilt-1M) hiyerarşik benzerlik araması ile geri çağrılır[cite: 65, 179].
* [cite_start]**KARG (Clinical Entity Extraction):** Geri çağrılan referans raporlarından klinik olarak anlamlı varlıklar yapılandırılmış bir formata çıkarılır[cite: 66, 181].
* [cite_start]**RAAF (Entity-Guided Explainability):** Çıkarılan klinik varlıkları WSI'nin ilgili bölgelerine ve destekleyici referans metinlerine bağlayan görsel açıklamalar üretilir[cite: 70, 183].
* [cite_start]**CSAL (Report Generation):** HL7 FHIR DiagnosticReport standartlarına uygun, görsel kanıtlara doğrudan bağlı standartlaştırılmış bir klinik özet üretilir[cite: 67, 68, 185].

## Veri Setleri (Datasets)
* [cite_start]**TCGA-BRCA:** CMEA ve RAAF modüllerinin eğitimi ve genel testler için kullanılan ~1.100 slaytlık temel veri seti[cite: 30, 293].
* [cite_start]**Quilt-1M:** Çok modlu referans veri tabanı aramaları için histopatoloji görüntüleri ve metinleri[cite: 30, 145].
* [cite_start]**CAMELYON16:** Açıklanabilirlik (RAAF) modülünün ve görsel geri çağırma performansının piksel düzeyinde doğrulanması için kullanılan 400 slaytlık yüksek kaliteli veri seti[cite: 30, 147, 294].

## Sistem Gereksinimleri ve Kısıtlamalar
### Mühendislik Kısıtlamaları (Design Constraints)
* [cite_start]**Donanım:** Geliştirme ve test süreçleri tek bir NVIDIA RTX 4090 (24GB VRAM) GPU, AMD Ryzen 9 7950X CPU, 64GB DDR5 RAM ve 2TB NVMe Gen4 SSD üzerinde çalışacak şekilde sınırlandırılmıştır[cite: 154, 283, 284, 285, 286].
* [cite_start]**Yazılım:** PyTorch 2.4, CUDA 12.1, OpenSlide, CLAM ve HuggingFace Transformers[cite: 288, 289, 290].
* [cite_start]**Veri Yönetimi:** Sisteme yüklenen dosyalar anonimleştirilmiş WSI dosyaları olmalıdır ve korunan sağlık bilgileri (KVKK ve GDPR uyumluluğu) işlenmeyecek veya saklanmayacaktır[cite: 163, 164, 169].
* [cite_start]**Zayıf Etiketleme:** Eğitim verisinde piksel düzeyinde tümör açıklamaları bulunmadığından, RAAF modülü "Multiple Instance Learning" ortamında çalışır[cite: 161, 162].

### Hedefler ve Değerlendirme Metrikleri (Goals and Evaluation Criteria)

| Hedef | Metrik | Hedef Değer | Değerlendirme Veri Seti |
| :--- | :--- | :--- | :--- |
| [cite_start]Doğru Geri Çağırma (Accurate Retrieval) [cite: 200] | [cite_start]Recall@5 [cite: 200] | > [cite_start]0.75 [cite: 200] | [cite_start]TCGA-BRCA (Held-out Test) [cite: 200] |
| [cite_start]Yüksek Kaliteli Sıralama (High-Quality Ranking) [cite: 200] [cite_start]| mAP (Mean Average Precision) [cite: 200] | > [cite_start]0.60 [cite: 200] | [cite_start]TCGA-BRCA (Held-out Test) [cite: 200] |
| [cite_start]Açıklanabilirlik (Explainability) [cite: 200] | [cite_start]IoU (Attention Mask vs. GT Mask) [cite: 200] | > [cite_start]0.40 [cite: 200] | [cite_start]CAMELYON16 (Test Set) [cite: 200] |
| [cite_start]Klinik Gerçeklik (Clinical Factuality) [cite: 200] | [cite_start]CEMS (Clinical Entity Matching Score) [cite: 200] | > [cite_start]0.85 [cite: 200] | [cite_start]TCGA-BRCA (Manual Review) [cite: 200] |
| [cite_start]Stil Tutarlılığı (Style Consistency) [cite: 201] | [cite_start]Style Divergence (KL Divergence) [cite: 201] | [cite_start]< 0.20 [cite: 201] | [cite_start]TCGA-BRCA [cite: 201] |
| [cite_start]Hesapsal Fizibilite (Computational Feasibility) [cite: 201] | [cite_start]WSI başına ortalama çıkarım süresi [cite: 201] | [cite_start]≤ 150 saniye [cite: 201] | [cite_start]TCGA-BRCA [cite: 201] |
| [cite_start]Kanıt İzlenebilirliği (Evidence Traceability) [cite: 201] | [cite_start]Kaynak linkine sahip varlıkların yüzdesi [cite: 201] | [cite_start]≥ %95 [cite: 201] | [cite_start]Manual Review [cite: 201] |

## Kullanım Senaryoları (Use Cases)
* [cite_start]**Senaryo 1: Yapay Zeka Destekli Otomatik Raporlama:** Kıdemli bir patolog, karmaşık bir vakayı sisteme yükler[cite: 228, 229, 230]. [cite_start]Sistem benzer vakaları bulur, nekroz bölgelerini ısı haritasıyla vurgular ve taslak bir rapor üretir[cite: 237, 239, 240]. [cite_start]Patolog raporu doğrulayıp 2 dakikadan kısa sürede onaylar[cite: 242].
* [cite_start]**Senaryo 2: Nadir Alt Türler İçin Teşhis Desteği:** Bir patoloji asistanı, şüphelendiği morfolojik bölgeyi seçerek tarihsel veritabanından benzer 5 vakayı çağırır ve hipotezini görsel olarak doğrular[cite: 244, 245, 248, 251, 254].
* [cite_start]**Senaryo 3: Kalite Güvencesi İçin Açıklanabilirlik Denetimi:** Laboratuvar yöneticisi, modelin rastgele artefaktlar üzerinden karar vermediğini doğrulamak için IoU maskeleri üzerinden sistemin dikkat odaklarını denetler[cite: 255, 256, 257, 263, 264].
* [cite_start]**Senaryo 4: Düşük Güven Protokolü:** Sistem yeterli benzerlikte vaka bulamadığında otonom rapor üretimini durdurur ve manuel inceleme önererek güvenliği sağlar[cite: 267, 271, 272, 274].

## Proje Planı ve İş Paketleri (Project Plan & Responsibilities)
* [cite_start]Proje 5 ana iş paketine (WP) bölünmüştür ve iki ekip üyesi arasında dağıtılmıştır[cite: 298, 299].
* [cite_start]**Emir Arda Eker (Language & Integration Lead):** WP-2 (Klinik raporların ayrıştırılması ve varlık çıkarımı), WP-4 (CSAL stil aktarımı ince ayarı) ve WP-5 (Web arayüzü ve API entegrasyonu) görevlerinden sorumludur[cite: 300, 301, 302, 303].
* [cite_start]**Faruk Rıza Öz (Vision & Architecture Lead):** WP-1 (WSI ön işleme ve UNI özellik çıkarımı), WP-3 (CMEA ve RAAF modüllerinin PyTorch uygulaması) ve CAMELYON16 üzerindeki doğrulama testlerinden sorumludur[cite: 304, 305, 306, 307].