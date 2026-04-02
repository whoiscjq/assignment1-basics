import pickle
import os
import pathlib
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable,Iterator
import regex as re

class Tokenizer:

    def __init__(self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab=vocab
        self.reverse_vocab={v:k for k,v in vocab.items()}
        self.merges=merges
        if special_tokens is None:
            self.special_tokens=[]
        else:
            self.special_tokens=sorted(special_tokens,key=lambda x:len(x),reverse=True) #sorting is necessary, because we want tokenizer to get longest special tokens first

        max_id=max(vocab.keys())
        for special_token in self.special_tokens:
            if self.reverse_vocab.get(special_token.encode("utf-8")) is None:
                max_id+=1
                self.vocab[max_id]=special_token.encode("utf-8")
                self.reverse_vocab[special_token.encode("utf-8")]=max_id

    @classmethod
    def from_files(
        cls, 
        vocab_filepath : str, 
        merges_filepath : str , 
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as f:  # 必须以二进制模式 'rb' 打开
            vocab= pickle.load(f)
        with open(merges_filepath, "rb") as f:  # 必须以二进制模式 'rb' 打开
            merges= pickle.load(f)

        return cls(vocab,merges,special_tokens)

    

    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        output=[]
        #find out special tokens
        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            pattern = "|".join(f"({token})" for token in escaped_tokens)
            docs = [doc for doc in re.split(pattern, text) if doc is not None]
        else:
            docs=[text]
        
        for doc in docs:
            
            # avoid special tokens split by the tokenizer
            if doc in self.special_tokens:
                output+=[self.reverse_vocab[doc.encode("utf-8")]]
            else:
                for word in re.finditer(PAT, doc):
                    word_str=word.group()
                    word_bytes=word_str.encode("utf-8")
                    word_list=list([bytes([letter]) for letter in list(word_bytes)])
                    idx=0
                    find_flag=True

                    #stop when there are not matched pair in the word_list
                    while find_flag: 
                        find_flag=False
                        for pair in self.merges:
                            idx=0
                            
                            #check whether merged_pair match and update word_list
                            while idx< len(word_list)-1 :
                                if (word_list[idx],word_list[idx+1]) == pair:
                                    merged_pair=b"".join((word_list[idx],word_list[idx+1]))
                                    del word_list[idx+1]
                                    del word_list[idx]
                                    word_list.insert(idx,merged_pair)
                                    find_flag=True
                                    break 

                                else:
                                    idx+=1
                            if find_flag==True:
                                break
                    encoded_word_list=list(map(lambda x: self.reverse_vocab[x], word_list))
                    output+=encoded_word_list
            
        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        
        for chunk in iterable:
            yield from self.encode(chunk)
            

    def decode(self, ids: list[int]) -> str:
        bytes_ids=list(map(lambda x: self.vocab.get(x), ids))
        return b"".join(bytes_ids).decode(encoding="utf-8", errors="replace")


TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
TRAIN_TXT_DATA_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
VAL_TXT_DATA_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.dat")
VAL_DATA_PATH = os.path.join(DATA_DIR, "valid.dat")

special_tokens = ["<|endoftext|>"]

# 读取词表和merges
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)
with open(MERGES_PATH, 'rb') as f:
    merges = pickle.load(f)

# 构造tokenizer
tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=special_tokens
)

print("=== 测试 Tokenizer ===")
test_texts = [
    "Once upon a time, there was a little robot.",
    "Hello world! <|endoftext|> Some more text.",
    "<|endoftext|>",
    "你好，世界！"
]

for text in test_texts:
    print(f"\n原文: {text}")
    encoded = tokenizer.encode(text)
    print("编码:", encoded)

    byte_tokens = [tokenizer.vocab[token_id] for token_id in encoded]
    str_tokens = [b.decode("utf-8", errors="replace") for b in byte_tokens]
    print("分词（可读）:", str_tokens)

    decoded = tokenizer.decode(encoded)
    print("解码:", decoded)
    print("是否完全还原:", decoded == text)



def encode_txt_as_numpy_array(tokenizer, path_to_txt, save_path):
    with open(path_to_txt, 'r') as f:
        num_lines = sum(1 for _ in f)
    
    # 第一步：统计总token数（需要遍历一遍）
    total_tokens = 0
    with open(path_to_txt, 'r') as f:
        for line in tqdm(f, total=num_lines, desc="Counting tokens"):
            total_tokens += len(tokenizer.encode(line))

    # 第二步：创建memmap
    dtype = np.int32
    tokens_mm = np.memmap(save_path, dtype=dtype, mode='w+', shape=(total_tokens,))

    # 第三步：再次遍历写入
    pos = 0
    with open(path_to_txt, 'r') as f:
        for line in tqdm(f, total=num_lines, desc="Tokenizing"):
            ids = tokenizer.encode(line)
            n = len(ids)
            tokens_mm[pos:pos+n] = ids
            pos += n

    tokens_mm.flush()

def main():
    encode_txt_as_numpy_array(tokenizer, TRAIN_TXT_DATA_PATH, TRAIN_DATA_PATH)
    encode_txt_as_numpy_array(tokenizer, VAL_TXT_DATA_PATH, VAL_DATA_PATH)


if __name__ == "__main__":
    main()