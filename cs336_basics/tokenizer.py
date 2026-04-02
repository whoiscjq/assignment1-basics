import pickle
import regex as re
from typing import Iterable,Iterator
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

if __name__=="__main__":
    text="Modern language models are typically trained with more sophisticated optimizers <|endoftext|>"
    vocab_filepath="tokenizer/tinystories_bpe_merges.pkl"
    merges_filepath="tokenizer/tinystories_bpe_merges.pkl"
    tokenizer=Tokenizer.from_files(vocab_filepath=vocab_filepath,merges_filepath=merges_filepath,special_tokens=["<|endoftext|>"])
    encoding=tokenizer.encode(text)
    print(encoding)
    decoding=tokenizer.decode(encoding)
    print(decoding)
    print(decoding==text)
    encoder_iterator=tokenizer.encode_iterable(text)
    # for idx in encoder_iterator:
    #     print(idx)

