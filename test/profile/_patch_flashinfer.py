path = '/home/songjq/.local/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/utils.cuh'
with open(path) as f:
    content = f.read()
old = "  } else if (group_size == 4) {"
idx8 = content.find("} else if (group_size == 8) {")
idx4 = content.find(old)
assert idx4 >= 0 and idx8 >= 0 and idx4 < idx8, (idx4, idx8)
six = """  } else if (group_size == 6) {                              \\
    constexpr size_t GROUP_SIZE = 6;                         \\
    __VA_ARGS__                                              \\
"""
content = content[:idx8] + six + content[idx8:]
with open(path, 'w') as f:
    f.write(content)
print("patched: inserted group_size==6")
