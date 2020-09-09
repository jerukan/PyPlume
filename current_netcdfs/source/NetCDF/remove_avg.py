from pathlib import Path

p = Path("")
for f in p.glob("*average*"):
	f.unlink()
