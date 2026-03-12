import gzip
import os
import pickle
import re
import math
import random
from collections import Counter
from pathlib import Path

class VietnameseBigramLM:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.vocab = set()
        self.V = 0

    def normalize_text(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize_syllables(self, sentence: str):
        sentence = self.normalize_text(sentence)

        # Giữ chữ cái tiếng Việt, số và khoảng trắng
        sentence = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", sentence, flags=re.UNICODE)
        sentence = re.sub(r"\s+", " ", sentence).strip()

        if not sentence:
            return []
        return sentence.split()

    def train_from_sentences(self, sentences):
        for sent in sentences:
            tokens = self.tokenize_syllables(sent)
            if not tokens:
                continue

            tokens = ["<s>"] + tokens + ["</s>"]

            for tok in tokens:
                self.unigram_counts[tok] += 1
                self.vocab.add(tok)

            for i in range(1, len(tokens)):
                self.bigram_counts[(tokens[i - 1], tokens[i])] += 1

        self.V = len(self.vocab)

    def bigram_prob(self, w_prev, w_curr):
        # Laplace smoothing
        return (self.bigram_counts[(w_prev, w_curr)] + 1) / (
            self.unigram_counts[w_prev] + self.V
        )

    def sentence_prob(self, sentence: str):
        tokens = ["<s>"] + self.tokenize_syllables(sentence) + ["</s>"]
        prob = 1.0
        for i in range(1, len(tokens)):
            prob *= self.bigram_prob(tokens[i - 1], tokens[i])
        return prob

    def sentence_log_prob(self, sentence: str):
        tokens = ["<s>"] + self.tokenize_syllables(sentence) + ["</s>"]
        logp = 0.0
        for i in range(1, len(tokens)):
            logp += math.log(self.bigram_prob(tokens[i - 1], tokens[i]))
        return logp

    def print_bigram_details(self, sentence: str):
        tokens = ["<s>"] + self.tokenize_syllables(sentence) + ["</s>"]
        print("\nChi tiết xác suất từng bigram:")
        for i in range(1, len(tokens)):
            prev_w = tokens[i - 1]
            curr_w = tokens[i]
            p = self.bigram_prob(prev_w, curr_w)
            print(f"P({curr_w} | {prev_w}) = {p:.10f}")

    def next_candidates(self, w_prev, top_k=10):
        candidates = []
        for w in self.vocab:
            if w == "<s>":
                continue
            p = self.bigram_prob(w_prev, w)
            candidates.append((w, p))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    import random

    def generate_sentence(self, max_len=15, top_k=10):
        current = "<s>"
        result = []

        for _ in range(max_len):
            candidates = []

            # Chỉ lấy những từ thực sự từng đi sau current
            for (prev_w, next_w), count in self.bigram_counts.items():
                if prev_w == current and next_w != "<s>":
                    candidates.append((next_w, count))

            if not candidates:
                break

            # Sắp xếp theo tần suất giảm dần
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Chỉ giữ top_k từ tốt nhất
            candidates = candidates[:top_k]

            words = [w for w, _ in candidates]
            probs = [self.bigram_prob(current, w) for w in words]

            # Chọn ngẫu nhiên có trọng số
            next_word = random.choices(words, weights=probs, k=1)[0]

            if next_word == "</s>":
                break

            # Tránh lặp 1 từ quá nhiều lần liên tiếp
            if len(result) >= 2 and result[-1] == next_word and result[-2] == next_word:
                continue

            result.append(next_word)
            current = next_word

        sentence = " ".join(result).strip()

        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        return sentence


def extract_corpus_from_titles_gz(input_gz: str, output_txt: str):
    count = 0
    kept = 0

    with gzip.open(input_gz, "rt", encoding="utf-8", errors="ignore") as fin, \
         open(output_txt, "w", encoding="utf-8") as fout:

        for line in fin:
            count += 1
            title = line.strip()

            if not title:
                continue

            # Loại một số title không hữu ích
            if ":" in title:
                # ví dụ: Thể loại:, Tập tin:, Thành viên:, Wikipedia:, ...
                continue

            # Loại title quá ngắn hoặc toàn số/ký tự lạ
            cleaned = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", title, flags=re.UNICODE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            if len(cleaned) < 2:
                continue

            fout.write(cleaned + "\n")
            kept += 1

    print(f"Tổng số dòng đọc từ dump: {count}")
    print(f"Số dòng giữ lại làm corpus: {kept}")
    print(f"Đã ghi ra file: {output_txt}")


def load_sentences_from_txt(path: str):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                sentences.append(s)
    return sentences


def merge_txt_files():
    folder = r".\Doi song"

    output_file = os.path.join(folder, "doi_song.txt")

    # 1. Xóa file cũ
    if os.path.exists(output_file):
        os.remove(output_file)

    txt_files = sorted(Path(folder).glob("*.txt"))

    # 2. Ghép file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file in txt_files:

            if file.name == "doi_song.txt":
                continue

            content = read_file_auto_encoding(file)

            outfile.write(content)
            outfile.write("\n")

    print("Đã tạo:", output_file)

def read_file_auto_encoding(file_path):
    encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1258", "latin1"]

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except:
            continue

    print("Không đọc được:", file_path)
    return ""

def main():
    # chạy
    merge_txt_files()

    corpus_txt = r".\Doi song\doi_song.txt"



    # print("\n=== ĐỌC CORPUS ===")


    model_file = "bigram_model.pkl"

    if os.path.exists(model_file):

        print("Loading model...")
        with open(model_file, "rb") as f:
            lm = pickle.load(f)

    else:

        print("Training model...")
        lm = VietnameseBigramLM()
        sentences = load_sentences_from_txt(corpus_txt)
        print(f"Số câu/tiêu đề dùng để train: {len(sentences)}")
        lm.train_from_sentences(sentences)

        with open(model_file, "wb") as f:
            pickle.dump(lm, f)

        print("Model saved.")

    print(f"Số unigram: {sum(lm.unigram_counts.values())}")
    print(f"Số bigram: {sum(lm.bigram_counts.values())}")
    print(f"Kích thước từ vựng |V|: {lm.V}")

    print("\n=== TÍNH XÁC SUẤT CÂU ===")
    test_sentence = "Hôm nay trời đẹp lắm"
    prob = lm.sentence_prob(test_sentence)
    log_prob = lm.sentence_log_prob(test_sentence)

    print(f'Câu kiểm tra: "{test_sentence}"')
    print(f"Xác suất câu: {prob:.20e}")
    print(f"Log xác suất: {log_prob:.10f}")
    lm.print_bigram_details(test_sentence)

    print("\n=== BƯỚC 6: SINH CÂU TỪ MÔ HÌNH ===")
    for i in range(10):
        print(f"{i+1}. {lm.generate_sentence(max_len=10)}")


if __name__ == "__main__":
    main()