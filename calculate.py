N=48
d_model=1600
d_ff=6400
vocab_size=50257

Total= N*(4*d_model*d_model+2*d_ff*d_model)+ 2*vocab_size*d_model
print(Total)
print(Total*4/1000000)