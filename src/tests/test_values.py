from src.value import Value

v1 = Value(10.0)
print(v1)

v2 = Value(10.199)
print(v2)

v3 = Value(35778)
print(v3)

v4 = Value("123hello")
print(v4)

v5 = Value("123")
print(v5)

v6 = Value(0)
print(v6)

v7 = Value(-112)

print(v7.round())