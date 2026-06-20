# TinyGPT - Budaya Jawa Corpus

## Deskripsi Proyek

Proyek ini merupakan implementasi model TinyGPT berbasis Transformer yang dilatih menggunakan corpus bertema Budaya Jawa. Corpus dibuat untuk memenuhi tugas pemrosesan bahasa alami (Natural Language Processing) dengan jumlah lebih dari 2000 kata.

Model dilatih menggunakan beberapa pendekatan tokenisasi menggunakan SentencePiece, yaitu:

* Character Tokenization (char)
* Word Tokenization (word)
* Unigram Tokenization (unigram)
* Byte Pair Encoding (BPE)

Tujuan proyek ini adalah membandingkan pengaruh berbagai metode tokenisasi terhadap performa model bahasa skala kecil (TinyGPT).

## Topik Corpus

Corpus berisi materi mengenai:

* Budaya Jawa
* Wayang Kulit
* Gamelan
* Rumah Adat Joglo
* Tari Tradisional Jawa
* Keraton Yogyakarta
* Upacara Adat Jawa
* Filosofi Kehidupan Masyarakat Jawa
* Batik Jawa
* Tradisi Sekaten

## Struktur Repository

```text
.
├── corpus.txt
├── tinygpt.py
├── transformer_blocks.py
├── tokenizer.model
├── tokenizer.vocab
└── README.md
```

## Teknologi yang Digunakan

* Python
* PyTorch
* SentencePiece
* Transformer Architecture

## Tujuan Eksperimen

1. Membuat corpus bertema Budaya Jawa dengan minimal 2000 kata.
2. Melatih model TinyGPT menggunakan corpus tersebut.
3. Membandingkan beberapa metode tokenisasi.
4. Menganalisis performa model berdasarkan nilai loss dan kualitas teks yang dihasilkan.

## Author

Nama: Irene Astri Anggraini Tonbesi

NIM: 23.11.5764

Universitas: Universitas Amikom Yogyakarta
