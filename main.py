with open("book.txt", "r", encoding="utf-8")as f:
    text = f.read()

chars = sorted(set(text))

string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join(int_to_string[i] for i in l) 

