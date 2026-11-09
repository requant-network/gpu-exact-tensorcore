"""Independent integer oracle and bounded signed encodings for exact GPU GEMM."""
FP4_DIGITS=(0,1,2,3,4,-8,6,-6,8,-4,-3,-2,-1)

def digits(value,base):
    result=[]
    while value:
        digit=FP4_DIGITS[value%13] if base==13 else (value+base//2)%base-base//2
        result.append(digit);value=(value-digit)//base
    return result or [0]

def reconstruct(values,base):return sum(d*base**i for i,d in enumerate(values))

def multiply(a,b,p):
    assert a and b and len(a[0])==len(b)
    return [[sum(x*y for x,y in zip(row,column))%p for column in zip(*b)] for row in a]

def bound(k,base):
    largest=8 if base==13 else base//2
    return k*largest*largest

def fp4_code(digit):
    magnitude={0:0,1:1,2:2,3:3,4:4,6:5,8:6,12:7}
    return magnitude[abs(digit)]|(8 if digit<0 else 0)
