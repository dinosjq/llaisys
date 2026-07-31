#!/usr/bin/env python3
"""Quick check: what GEMM/GEMV kernels ran in decode phase?"""
import sqlite3, sys

conn = sqlite3.connect(sys.argv[1])
cur = conn.cursor()

# Find prefill end
cur.execute("""SELECT MAX(e2.end) FROM CUPTI_ACTIVITY_KIND_KERNEL e2
    JOIN StringIds s2 ON s2.id = e2.demangledName
    WHERE s2.value LIKE '%paged_attention_kernel%'""")
prefill_end = cur.fetchone()[0]

# Decode kernels
cur.execute("""SELECT s.value, COUNT(*) as cnt, SUM(e.end - e.start) as tot
    FROM CUPTI_ACTIVITY_KIND_KERNEL e
    JOIN StringIds s ON s.id = e.demangledName
    WHERE e.start > ? AND (s.value LIKE '%gemv%' OR s.value LIKE '%gemm%')
    GROUP BY s.value ORDER BY cnt DESC LIMIT 20""", (prefill_end,))
for r in cur.fetchall():
    print(f'{r[1]:5d}  {r[2]/1e6:8.2f}ms  {r[0][:130]}')
conn.close()
